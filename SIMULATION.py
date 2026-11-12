import csv      # Saves experimental data to CSV files
import time     # Real-time delays when viewing the simulation
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are     # For solving the Riccati equation
from PIL import Image, ImageDraw    # Saves images from the RGB camera and draws the detection marker on them
import matplotlib   # Creates the graphs and plots
matplotlib.use("Agg")   # Want to save plots to image files instead of opening MPL
import matplotlib.pyplot as plt
from scene_setup import reset_scene     # Calls the reset scene function
from vision import (WristCamera, detect_ball_camera, R_MJ_TO_OPT)    # Captures RGB-D images and detects the ball
from gripper import fingers_touching    # Determines whether the gripper fingers contact the ball
from Camera_calibration import calibrate    # Calibrates the wrist camera using a set of ball positions


# True runs 30 trials (ten positions, three conditions)
# False runs one pick, with optional viewer and torque disturbance
ROBUSTNESS_TEST = False

SHOW_VIEWER = True      # Show the MuJoCo window in a single run
WRIST_VIEW = False       # True shows the wrist camera instead of the outside view

# Apply the same external torque during a single demonstration.
DEMO_DISTURBANCE = True

# Ball position for a single demonstration
SEED = 0

# Ball positions used for robustness testing
TEST_SEEDS = range(10)

# Test the same ten ball positions under three different conditions
# The noisy condition has 1.5 times the baseline encoder noise
CONDITIONS = {
    "baseline": {"encoder_std": 0.005, "torque_Nm": 0.0},

    "noisy": {"encoder_std": 0.0075, "torque_Nm": 0.0},

    "disturbed": {"encoder_std": 0.005, "torque_Nm": 0.5}
}

# Apply a short torque pulse to the wrist during the approach
DISTURBANCE_START = 2.0
DISTURBANCE_END = 2.2

# Apply it about the positive world Y axis
DISTURBANCE_DIRECTION = np.array([0.0, 1.0, 0.0])

DT = 0.01                     # Controller update interval: 0.01 s (100 Hz)
MAX_TIME = 55.0               # Maximum simulation time per trial
CAMERA_EVERY = 25             # Capture a new RGB-D image every 25 control steps

APPROACH_HEIGHT = 0.12        # First move the fingers 12 cm above the ball
GRASP_HEIGHT = 0.018          # Then aim 18 mm above the detected ball centre
LIFT_HEIGHT = 0.10            # Raise the gripped ball by 10 cm

MAX_SPEED = 0.25              # Maximum desired joint speed (rad/s)
MAX_LAG = np.deg2rad(3)       # Keep reference angles within 3 degrees of estimates

FOLDER = Path(__file__).resolve().parent  # Find scene.xml beside this script
RESULTS = FOLDER / "results" / "pick_evidence"  # Save graphs and CSVs here


"""
Calibrates the wrist camera once, before any picks run
"""
def calibrate_camera():
    model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))   # Separate copy of the scene
    data = mujoco.MjData(model)
    camera = WristCamera(model)
    camera_matrix, _, _ = calibrate(model, data, camera)    # Keep K, ignore the error values
    camera.renderer.close()
    print("Calibrated camera matrix:\n", camera_matrix.round(2), flush=True)
    return camera_matrix


"""
Runs one pick using the chosen ball position, encoder noise, and torque
Returns the trial results, saving detailed evidence for selected trials
"""
def run_trial(
    seed,
    condition,
    encoder_std,
    torque_Nm,
    show_viewer=False
):

    output = RESULTS / condition / f"seed_{seed}"

    # Save per-timestep evidence for single demonstrations and seed 0 of each batch
    # The other 27 batch runs contribute only to the shared summary CSV
    save_evidence = (not ROBUSTNESS_TEST) or seed == 0
    if save_evidence:
        output.mkdir(parents=True, exist_ok=True)
        first_photo = output / "wrist_camera_first_detection.png"
        if first_photo.exists():
            first_photo.unlink()        # Avoids reusing an old camera screenshot

    print(f"\n=== {condition.upper()} | seed {seed} ===", flush=True)

    # Chloe's reset function places the robot in its look pose
    # and randomly positions the ball using the supplied seed.
    model = mujoco.MjModel.from_xml_path(str(FOLDER / "scene.xml"))

    # Create a fresh simulation state so each seed starts independently.
    data = mujoco.MjData(model)

    # Reset the arm pose and place the ball according to this seed.
    reset_scene(model, data, seed)

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint"
    ]

    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3"
    ]

    # Finds where MuJoCo stores the six joint positions
    q_id = np.array([model.joint(name).qposadr[0] for name in joint_names])

    # Finds where MuJoCo stores the six joint velocities
    v_id = np.array([model.joint(name).dofadr[0] for name in joint_names])

    # Maps motor names onto the actuator indices used for torque commands
    motor_id = np.array([model.actuator(name).id for name in motor_names])

    # Maps each encoder name onto its sensor-data index
    encoder_id = np.array([model.sensor("encoder_" + name).adr[0] for name in joint_names])

    gripper_id = model.actuator("gripper_fingers_actuator").id
    site_id = model.site("attachment_site").id
    camera_id = model.camera("wrist_camera").id
    ball_id = model.body("target_ball_body").id

    # Calculates the number of physics updates per controller step
    physics_steps = round(DT / model.opt.timestep)

    # Applies the external torque to the wrist_3_link 
    disturbance_body = model.body("wrist_3_link").id

    """STATE-SPACE MODEL AND LQR"""

    # State = [joint angle, joint angular velocity]
    # Input = joint angular acceleration
    # Measurement = encoder joint angle

    A = np.array([
        [1, DT],
        [0, 1]
    ])

    B = np.array([
        [0.5 * DT**2],
        [DT]
    ])

    C = np.array([[1, 0]])

    # Test controllability and observability
    controllability = np.hstack((B,A @ B))
    observability = np.vstack((C, C @ A))

    print("Controllability rank:", np.linalg.matrix_rank(controllability))
    print("Observability rank:", np.linalg.matrix_rank(observability))

    # LQR weighting matrices.
    Q = np.diag([100.0, 10.0])    # Penalise angle error and velocity error
    R = np.array([[1.0]])         # Penalise large acceleration commands

    # Solves the discrete Riccati equation for the LQR feedback law
    P_lqr = solve_discrete_are(A, B, Q, R)

    # Calculates the two feedback gains: angle error and velocity error
    K = np.linalg.solve(R + B.T @ P_lqr @ B, B.T @ P_lqr @ A).ravel()

    print("LQR gain:", K)

    """Kalman Filter"""
    # encoder_std comes from the selected experiment
    # The Kalman measurement covariance changes with it
    acceleration_std = 0.5

    # Encoder measurement covariance is variance = standard deviation squared
    R_noise = np.array([[encoder_std**2]])

    # Process covariance describes uncertainty in the acceleration model
    Q_noise = (acceleration_std**2 * (B @ B.T) + 1e-10 * np.eye(2))

    # One estimated angle and angular velocity for each of the six joints
    estimate = np.zeros((6, 2))

    # Initialise the angle estimates from encoders; velocity starts at zero
    estimate[:, 0] = data.sensordata[encoder_id]

    # Covariance matrices for all six Kalman filters
    P = np.repeat(np.diag([0.1**2, 0.5**2])[None], 6, axis=0)

    previous_acceleration = np.zeros(6)

    # Keep the random samples repeatable between trials
    rng = np.random.default_rng(42)

    # Separate data object for computing kinematics and dynamics from estimates
    estimated_data = mujoco.MjData(model)

    estimated_data.qpos[:] = model.qpos0
    estimated_data.qpos[q_id] = estimate[:, 0]

    mujoco.mj_forward(model, estimated_data)

    # Prepare the dense inertia matrix for computed torque control
    mass_matrix = np.zeros((model.nv, model.nv))

    """Camera and Gripper"""

    camera_matrix = CAMERA_MATRIX    # Calibrated once at the start of the run

    camera = WristCamera(model)

    # Calculate the midpoint between the open gripper pads
    site_R = estimated_data.site_xmat[site_id].reshape(3, 3).copy()
    site_position = estimated_data.site_xpos[site_id].copy()
    left_pad = estimated_data.body("gripper_left_pad").xpos
    right_pad = estimated_data.body("gripper_right_pad").xpos
    finger_midpoint = (left_pad + right_pad) / 2

    # Express the finger midpoint offset relative to the tool frame
    finger_offset = (site_R.T @ (finger_midpoint - site_position))

    # Maintain the original gripper orientation
    start_orientation = site_R.copy()

    """Movement Stages"""

    stage = "SEARCH"
    stage_start = data.time

    ball_estimate = None
    last_seen = -100.0

    q_target = estimate[:, 0].copy()
    qdot_target = np.zeros(6)

    gripper_command = 0.0
    touch_steps = 0

    lift_target = None

    stop_reason = "Maximum simulation time"

    # Ground truth for evaluating the final pick only
    ball_start = data.xpos[ball_id].copy()

    # Collect evaluation data separately from the controller commands
    records = []  # Time history is saved only for demonstration/representative trials

    angle_errors = []
    velocity_errors = []
    tracking_errors = []
    camera_errors = []

    max_torque = 0.0
    saturated_steps = 0

    disturbance_seen = False

    last_print = -1

    viewer = (mujoco.viewer.launch_passive(model, data) if show_viewer else None)

    if viewer is not None:

        if WRIST_VIEW:

            with viewer.lock(): #

                viewer.cam.type = (
                    mujoco.mjtCamera.mjCAMERA_FIXED
                )

                viewer.cam.fixedcamid = camera_id

        viewer.sync()
        time.sleep(1)

    print("Starting vision-guided pick...", flush=True)

    try:
        """MAIN CONTROL LOOP"""

        for step in range(int(MAX_TIME / DT)):

            old_stage = stage

            mujoco.mj_forward(model, data)

            """
            Kalman Filter
            """

            # Read the encoders and add noise; never use true qpos for control
            measurements = (data.sensordata[encoder_id].copy() + rng.normal(0, encoder_std, 6))

            for j in range(6):

                # Predict the angle and angular velocity
                x_previous = estimate[j].reshape(2, 1)

                x_pred = (A @ x_previous + B * previous_acceleration[j])

                # Predict how uncertain the state estimate has become
                P_pred = (A @ P[j] @ A.T + Q_noise)

                # Innovation is the difference between measured and predicted angle
                innovation = (measurements[j] - (C @ x_pred).item())

                S = (C @ P_pred @ C.T + R_noise)

                # Calculate how much the measurement should correct the prediction
                kalman_gain = np.linalg.solve(S, C @ P_pred).T

                # Correct the predicted angle and angular velocity
                x_corrected = (x_pred + kalman_gain * innovation)

                estimate[j] = x_corrected.flatten()

                # Update uncertainty.
                I = np.eye(2)

                # Joseph covariance update keeps the uncertainty matrix stable
                P[j] = (
                    (I - kalman_gain @ C)
                    @ P_pred
                    @ (I - kalman_gain @ C).T
                    + kalman_gain
                    @ R_noise
                    @ kalman_gain.T
                )

            # Calculate estimated robot kinematics
            estimated_data.qpos[q_id] = estimate[:, 0]

            estimated_data.qvel[:] = 0

            estimated_data.qvel[v_id] = estimate[:, 1]

            mujoco.mj_forward(model, estimated_data)

            current_R = estimated_data.site_xmat[site_id].reshape(3, 3)

            # Estimated world position of the midpoint between the fingers
            finger_position = (estimated_data.site_xpos[site_id] + current_R @ finger_offset)

            """RGB-D Camera"""

            if (stage in ("SEARCH", "APPROACH") and step % CAMERA_EVERY == 0):

                rgb, depth = camera.capture(data)

                point_camera, pixel = detect_ball_camera(rgb, depth, camera_matrix)

                # Only keep the first valid RGB-D detection for demonstration evidence
                # The camera is still captured and processed normally for every trial
                if save_evidence and point_camera is not None and not first_photo.exists():
                    picture = Image.fromarray(rgb)
                    if pixel is not None:
                        u, v = pixel
                        ImageDraw.Draw(picture).ellipse((u - 8, v - 8, u + 8, v + 8), outline="lime", width=3)
                    picture.save(first_photo)

                if point_camera is not None:
                    # Camera-to-world transformation
                    # Uses estimated camera pose
                    R_wc = (estimated_data.cam_xmat[camera_id].reshape(3, 3) @ R_MJ_TO_OPT)

                    camera_position = (estimated_data.cam_xpos[camera_id])

                    # Transform the ball from optical camera coordinates into world position
                    detected_ball = (camera_position + R_wc @ point_camera)

                    # Camera localisation error for EVALUATION
                    # This value is never used for planning
                    true_ball = data.xpos[ball_id].copy()

                    camera_errors.append(np.linalg.norm(detected_ball - true_ball) * 1000)

                    # Smooth the camera position measurement
                    if ball_estimate is None:
                        ball_estimate = detected_ball
                    else:
                        ball_estimate = (0.65 * ball_estimate + 0.35 * detected_ball)

                    last_seen = data.time

                    if stage == "SEARCH":
                        stage = "APPROACH"

            """PBVS Trajectory Planning"""

            goal = None

            if stage == "APPROACH":
                goal = (ball_estimate + np.array([0, 0, APPROACH_HEIGHT]))

            elif stage == "DESCEND":
                goal = (ball_estimate + np.array([0, 0, GRASP_HEIGHT]))

            elif stage == "LIFT":
                goal = lift_target

            if goal is not None:
                # Difference between desired and estimated finger position
                position_error = (goal - finger_position)

                distance = np.linalg.norm(position_error)

                # Small-angle orientation error keeps the gripper pointing down
                rotation_error = 0.5 * sum((np.cross(current_R[:, i], start_orientation[:, i]) for i in range(3)), np.zeros(3))

                # Desired Cartesian velocity
                desired_velocity = np.concatenate((0.65 * position_error, 0.55 * rotation_error))

                # Jacobian at the finger midpoint
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))

                mujoco.mj_jac(model, estimated_data, jacp, jacr, finger_position, model.site_bodyid[site_id])

                J = np.vstack((jacp[:, v_id], jacr[:, v_id]))

                # Damped least-squares inverse Jacobian
                damping = 0.06

                # Damped least squares converts Cartesian velocity to joint velocity
                qdot_target = (J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), desired_velocity))

                # Limit joint velocities
                peak_speed = np.max(np.abs(qdot_target))

                if peak_speed > MAX_SPEED:

                    qdot_target *= (MAX_SPEED / peak_speed)

                # Keep the desired position close to the estimated joint position
                q_target = np.clip(q_target, estimate[:, 0] - MAX_LAG, estimate[:, 0] + MAX_LAG)

                # Integrate desired joint velocity to update the joint position reference
                q_target += qdot_target * DT

                # Check whether the stage is complete
                if (stage == "APPROACH" and distance < 0.008):
                    stage = "DESCEND"

                elif (stage == "DESCEND" and distance < 0.005):
                    stage = "CLOSE"

                elif (stage == "LIFT" and distance < 0.025):
                    stage = "HOLD"

            else:
                # Hold current position when not moving
                qdot_target[:] = 0

                q_target = np.clip(q_target, estimate[:, 0] - MAX_LAG, estimate[:, 0] + MAX_LAG)

            """Gripper"""

            if stage == "CLOSE":
                gripper_command = min(255, gripper_command + 1.5)
                left_touch, right_touch = fingers_touching(model, data)
                if left_touch and right_touch:
                    touch_steps += 1
                    # Slightly increase grip after contact.
                    if touch_steps == 1:
                        gripper_command = min(255, gripper_command + 35)
                else:
                    touch_steps = 0

                # Both fingers must maintain contact
                if touch_steps >= 40:
                    lift_target = (finger_position + np.array([0, 0, LIFT_HEIGHT]))
                    stage = "LIFT"

            # Hold the object for 1.5 seconds
            if (old_stage == "HOLD" and data.time - stage_start >= 1.5):
                stage = "DONE"
                stop_reason = "Sequence completed"

            if stage != old_stage:
                stage_start = data.time
                if show_viewer:
                    print(
                        f"{data.time:.2f}s: "
                        f"{old_stage} -> {stage}",
                        flush=True
                    )

            """LGR and Computed Torque"""

            desired_states = np.column_stack((q_target, qdot_target))

            error = estimate - desired_states

            # LQR turns the estimated state error into desired acceleration
            acceleration = -(error @ K)

            # Robot inertia matrix
            mujoco.mj_fullM(model, estimated_data, mass_matrix)

            arm_mass = mass_matrix[np.ix_(v_id, v_id)]

            # Computed torque compensates for robot inertia and bias forces
            torque = (arm_mass @ acceleration + estimated_data.qfrc_bias[v_id])

            # Apply motor torque limits
            original_torque = torque.copy()

            torque = np.clip(torque, model.actuator_ctrlrange[motor_id, 0], model.actuator_ctrlrange[motor_id, 1])

            if np.any(torque != original_torque):
                saturated_steps += 1

            # Send the control inputs to the actual simulated arm and gripper
            data.ctrl[motor_id] = torque
            data.ctrl[gripper_id] = gripper_command

            previous_acceleration = acceleration.copy()

            """Physical Disturbance"""

            # Entries 3:6 are a torque in N m, expressed in world axes
            # It acts on the simulated wrist without being given to the observer
            disturbance_active = (torque_Nm > 0 and DISTURBANCE_START <= data.time < DISTURBANCE_END and stage == "APPROACH")

            # Reset the applied torque outside the disturbance interval
            data.xfrc_applied[disturbance_body, :] = 0

            if disturbance_active:
                data.xfrc_applied[disturbance_body, 3:6] = torque_Nm * DISTURBANCE_DIRECTION
                disturbance_seen = True

            """Recording the Results"""

            # Ground truth is used ONLY to calculate errors for evaluation
            # Neither the planner nor the controller reads these states
            true_q = data.qpos[q_id].copy()
            true_dq = data.qvel[v_id].copy()

            angle_errors.append(estimate[:, 0] - true_q)

            velocity_errors.append(estimate[:, 1] - true_dq)

            tracking_errors.append(q_target - true_q)

            max_torque = max(max_torque, float(np.max(np.abs(torque))))

            # True ball lift is an evaluation measurement
            ball_lift = 1000 * (data.xpos[ball_id, 2] - ball_start[2])

            # Record the true state, estimate and uncertainty for selected plots
            if save_evidence:
                records.append({
                    "time_s": float(data.time),
                    "stage": stage,
                    "external_torque_Nm": float(torque_Nm if disturbance_active else 0),
                    "ball_lift_mm": float(ball_lift),
                    "true_angle_deg": float(np.rad2deg(true_q[0])),
                    "estimated_angle_deg": float(np.rad2deg(estimate[0, 0])),
                    "angle_2sigma_deg": float(np.rad2deg(2 * np.sqrt(P[0, 0, 0]))),
                    "desired_angle_deg": float(np.rad2deg(q_target[0])),
                    "true_velocity_rad_s": float(true_dq[0]),
                    "estimated_velocity_rad_s": float(estimate[0, 1]),
                    "velocity_2sigma_rad_s": float(2 * np.sqrt(P[0, 1, 1])),
                    "max_tracking_error_deg": float(np.rad2deg(np.max(np.abs(q_target - true_q)))),
                    "max_torque_Nm": float(np.max(np.abs(torque)))
                })

            # Advance the physics simulation
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)

            if (viewer is not None and viewer.is_running()):
                viewer.sync()

            if show_viewer and int(data.time) != last_print:
                last_print = int(data.time)

                print(
                    f"{data.time:.1f}s | "
                    f"{stage:8s} | "
                    f"finger height: "
                    f"{finger_position[2]:.3f} m",
                    flush=True
                )

            if stage == "DONE":
                break

            if stage == "SEARCH" and data.time > 4:
                stop_reason = "Camera could not find the ball"
                break

            if (stage == "APPROACH" and data.time - last_seen > 2.0):
                stop_reason = "Camera lost the ball"
                break

            if (stage == "APPROACH" and data.time - stage_start > 24):
                stop_reason = "Approach timed out"
                break

            if (stage == "DESCEND" and data.time - stage_start > 12):
                stop_reason = "Descent timed out"
                break

            if (stage == "CLOSE" and data.time - stage_start > 4):
                stop_reason = "Grasp unsuccessful"
                break

            if (stage == "LIFT" and data.time - stage_start > 16):
                stop_reason = "Lift timed out"
                break

            if (not np.all(np.isfinite(torque)) or np.max(np.abs(q_target - estimate[:, 0])) > np.deg2rad(5)):
                stop_reason = "Controller error"
                break

            if (viewer is not None and viewer.is_running()):
                time.sleep(DT)

    finally:
        camera.renderer.close()
        if viewer is not None:
            print("Close the MuJoCo viewer to finish.")

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            viewer.close()

    """Final Results"""
    mujoco.mj_forward(model, data)

    ball_final = data.xpos[ball_id].copy()

    left_touch, right_touch = fingers_touching(model, data)

    lifted_mm = 1000 * (ball_final[2] - ball_start[2])

    # Require completion, at least 60 mm lift, and both fingers still touching
    success = (stage == "DONE" and lifted_mm > 60 and left_touch and right_touch)

    angle_errors = np.array(angle_errors)
    velocity_errors = np.array(velocity_errors)
    tracking_errors = np.array(tracking_errors)

    angle_rmse = float(np.rad2deg(np.sqrt(np.mean(angle_errors**2))))

    velocity_rmse = float(np.sqrt(np.mean(velocity_errors**2)))

    tracking_rmse = float(np.rad2deg(np.sqrt(np.mean(tracking_errors**2))))

    peak_tracking = float(np.rad2deg(np.max(np.abs(tracking_errors))))

    """
    Trial Data Export and Graphing (Largely AI generated)
    """

    # Save full time-history evidence only for demonstration or representative trials
    # The plot shows true state, Kalman estimate, uncertainty and joint tracking together
    if save_evidence and records:
        with (output / "trial_data.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        t = np.array([r["time_s"] for r in records])  # Simulation time
        true_angle = np.array([r["true_angle_deg"] for r in records])
        estimated_angle = np.array([r["estimated_angle_deg"] for r in records])
        angle_band = np.array([r["angle_2sigma_deg"] for r in records])
        true_velocity = np.array([r["true_velocity_rad_s"] for r in records])
        estimated_velocity = np.array([r["estimated_velocity_rad_s"] for r in records])
        velocity_band = np.array([r["velocity_2sigma_rad_s"] for r in records])
        tracking = np.array([r["max_tracking_error_deg"] for r in records])

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(t, true_angle, label="True angle")
        axes[0].plot(t, estimated_angle, "--", label="Kalman estimate")
        axes[0].fill_between(t, estimated_angle - angle_band, estimated_angle + angle_band, alpha=0.2, label="Estimate +/- 2 sigma")
        axes[0].set_ylabel("Shoulder angle (deg)")
        axes[1].plot(t, true_velocity, label="True velocity")
        axes[1].plot(t, estimated_velocity, "--", label="Kalman estimate")
        axes[1].fill_between(t, estimated_velocity - velocity_band, estimated_velocity + velocity_band, alpha=0.2, label="Estimate +/- 2 sigma")
        axes[1].set_ylabel("Shoulder speed (rad/s)")
        axes[2].plot(t, tracking, label="Maximum error across six joints")
        axes[2].set_ylabel("Tracking error (deg)")
        axes[2].set_xlabel("Simulation time (s)")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            if disturbance_seen:
                ax.axvspan(DISTURBANCE_START, DISTURBANCE_END, alpha=0.15, label="External torque applied")
            ax.legend(loc="best")

        fig.suptitle(f"{condition} | seed {seed}: state estimation and tracking")
        fig.tight_layout()
        fig.savefig(output / "trial_analysis.png", dpi=160)
        plt.close(fig)

    """Returns Trial Results"""

    camera_rmse = (float(np.sqrt(np.mean(np.square(camera_errors)))) if camera_errors else None)

    result = {
        "condition": condition,
        "seed": seed,
        "encoder_std_rad": encoder_std,
        "external_torque_Nm": torque_Nm,
        "disturbance_applied": int(disturbance_seen),
        "success": int(success),
        "final_stage": stage,
        "reason": stop_reason,
        "ball_x_m": float(ball_start[0]),
        "ball_y_m": float(ball_start[1]),
        "lift_mm": float(lifted_mm),
        "duration_s": float(data.time),
        "angle_rmse_deg": angle_rmse,
        "velocity_rmse_rad_s": velocity_rmse,
        "tracking_rmse_deg": tracking_rmse,
        "peak_tracking_deg": peak_tracking,
        "peak_motor_torque_Nm": max_torque,
        "saturated_steps": saturated_steps,
        "camera_rmse_mm": camera_rmse
    }

    print(
        f"\nResult: "
        f"{'SUCCESS' if success else 'FAILURE'}"
    )

    print(f"Final stage: {stage}")
    print(f"Reason: {stop_reason}")
    print(f"Ball lift: {lifted_mm:.1f} mm")
    print(f"External torque applied: {disturbance_seen}")
    print(f"Kalman angle RMSE: {angle_rmse:.3f} deg")
    print(f"Kalman velocity RMSE: {velocity_rmse:.4f} rad/s")
    print(f"Joint tracking RMSE: {tracking_rmse:.3f} deg")
    print(f"Maximum tracking error: {peak_tracking:.3f} deg")
    print(f"Maximum commanded motor torque: {max_torque:.3f} N m")
    if save_evidence:
        print(f"Detailed evidence saved to: {output}", flush=True)

    return result


"""Comparing Robustness Trials"""

def save_comparison(trials):

    RESULTS.mkdir(parents=True, exist_ok=True)

    # One CSV contains all results, including success, lift and error measurements
    with (RESULTS / "robustness_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    conditions = list(CONDITIONS.keys())
    seeds = list(TEST_SEEDS)
    report = []

    for condition in conditions:
        rows = [r for r in trials if r["condition"] == condition]
        successes = sum(r["success"] for r in rows)
        tracking = np.mean([r["tracking_rmse_deg"] for r in rows])
        angle = np.mean([r["angle_rmse_deg"] for r in rows])
        report.append(
            f"{condition}: {successes}/{len(rows)} picks "
            f"({100 * successes / len(rows):.0f}%), "
            f"tracking RMSE {tracking:.3f} deg, "
            f"Kalman angle RMSE {angle:.3f} deg"
        )
        failures = [r["seed"] for r in rows if not r["success"]]
        report.append(f"  Failed seeds: {failures}")

    report.append(
        f"Disturbance: {CONDITIONS['disturbed']['torque_Nm']} N m "
        f"about world +Y, from {DISTURBANCE_START} to {DISTURBANCE_END} s."
    )
    report.append("True joint/ball states are used for evaluation only.")
    report_text = "\n".join(report) + "\n"
    (RESULTS / "robustness_summary.txt").write_text(report_text)
    print("\n" + report_text, flush=True)

    # Compare state-estimation accuracy and trajectory tracking separately
    # Plot each seed for all three conditions so variability is visible
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    comparisons = [
        ("tracking_rmse_deg", "Joint tracking RMSE (deg)"),
        ("angle_rmse_deg", "Kalman angle RMSE (deg)")
    ]

    for ax, (field, ylabel) in zip(axes, comparisons):
        for condition in conditions:
            rows = [r for r in trials if r["condition"] == condition]
            rows.sort(key=lambda r: r["seed"])
            ax.plot(seeds, [r[field] for r in rows], "o-", label=condition)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[1].set_xticks(seeds)
    axes[1].set_xlabel("Ball randomisation seed")
    fig.suptitle("Tracking and estimation across robustness conditions")
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_errors.png", dpi=160)
    plt.close(fig)

    # Show that different seeds really produced different ball positions
    baseline = [r for r in trials if r["condition"] == "baseline"]
    plt.figure(figsize=(7, 5))

    for r in baseline:
        plt.scatter(r["ball_x_m"], r["ball_y_m"],
                    marker="o" if r["success"] else "x", s=75)
        plt.annotate(str(r["seed"]), (r["ball_x_m"], r["ball_y_m"]),
                     xytext=(5, 4), textcoords="offset points")

    plt.xlabel("Ball X (m)")
    plt.ylabel("Ball Y (m)")
    plt.title("Randomised ball positions (circle=success, cross=failure)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "ball_positions.png", dpi=160)
    plt.close()

    print(
        "Comparison plots saved to:",
        RESULTS,
        flush=True
    )


"""Actually Starts the Tests"""

if __name__ == "__main__":
    CAMERA_MATRIX = calibrate_camera()    # Runs once, then every trial uses it

    if ROBUSTNESS_TEST:
        print("ROBUSTNESS TEST: " "10 ball positions x 3 conditions = 30 trials", flush=True)

        trials = []

        for condition, settings in CONDITIONS.items():
            for seed in TEST_SEEDS:
                trial = run_trial(
                    seed=seed,
                    condition=condition,
                    encoder_std=settings["encoder_std"],
                    torque_Nm=settings["torque_Nm"],
                    show_viewer=False
                )
                trials.append(trial)

        save_comparison(trials)

    else:
        # Demonstrate the same 0.5 N m torque pulse
        disturbance_torque = (CONDITIONS["disturbed"]["torque_Nm"] if DEMO_DISTURBANCE else 0.0)

        condition = ("demo_disturbed" if DEMO_DISTURBANCE else "demo_baseline")

        run_trial(
            seed=SEED,
            condition=condition,
            encoder_std=0.005,
            torque_Nm=disturbance_torque,
            show_viewer=SHOW_VIEWER
        )
