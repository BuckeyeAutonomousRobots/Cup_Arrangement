"""Predeclared one-demo memorization metrics, not generalization or task success."""
import numpy as np
from position_act_contract import adapter_metrics


def evaluate(position, q, commands, side, ticks):
    valid = side['valid']
    target = side['targets']
    velocity = (position[:, :6] - q[:, :6]) / .1
    desired = (target[:, :6] - q[:, :6]) / .1
    moving = valid[:, None] & (np.abs(desired) > .02)
    moving_error = float(np.abs(velocity - desired)[moving].mean())
    identity_error = float(np.abs(desired)[moving].mean())
    hold = valid & (np.max(np.abs(desired), axis=1) <= .005)
    closed = valid & (commands[:, 6] < -.004)
    opened = valid & (commands[:, 6] > -.0001)
    closed_motion = moving & closed[:, None]
    report = adapter_metrics(position, q, commands, target, ticks, valid)
    report.update(
        moving_elements=int(moving.sum()), moving_raw_velocity_mae=moving_error,
        identity_moving_raw_velocity_mae=identity_error,
        moving_error_reduction_vs_identity=1 - moving_error / identity_error,
        all_arm_hold_samples=int(hold.sum()),
        all_arm_hold_false_fraction=float(np.mean(np.max(np.abs(velocity[hold]), axis=1) > .02)) if hold.any() else None,
        all_arm_hold_mean_abs_velocity=float(np.abs(velocity[hold]).mean()) if hold.any() else None,
        closed_jaw_moving_elements=int(closed_motion.sum()),
        closed_jaw_moving_raw_velocity_mae=float(np.abs(velocity - desired)[closed_motion].mean()) if closed_motion.any() else None,
        jaw_mae_m=float(np.abs(position[valid, 6] - commands[valid, 6]).mean()),
        strong_close_samples=int(closed.sum()),
        strong_close_recall=float(np.mean(position[closed, 6] < -.004)) if closed.any() else None,
        open_samples=int(opened.sum()),
        open_recall=float(np.mean(np.abs(position[opened, 6]) <= .0002)) if opened.any() else None,
    )
    return report


def gates(report, reference):
    return dict(
        moving_error_reduction=report['moving_error_reduction_vs_identity'] >= .9,
        stationary_initiation=report['stationary_samples'] > 0 and report['stationary_correct'] == report['stationary_samples'],
        rotation_recall=report['rotation_samples'] > 0 and report['rotation_correct'] / report['rotation_samples'] >= .95,
        all_arm_holding=report['all_arm_hold_false_fraction'] is not None and report['all_arm_hold_false_fraction'] <= .01,
        wrist_steady_holding=report['steady_samples'] > 0 and report['steady_false'] / report['steady_samples'] <= .005,
        jaw_accuracy=report['jaw_mae_m'] <= .0002,
        closure=report['strong_close_recall'] is not None and report['strong_close_recall'] >= .95,
        release=report['open_recall'] is not None and report['open_recall'] >= .95,
        tracking_guard=report['tracking_guard_violations'] == 0,
        # Measured-next-position reference itself slightly exceeds the command
        # cap on some ticks. Compare excess saturation to that fixed reference;
        # the actual adapter command limit remains +/-0.245 rad/s unchanged.
        excess_adapter_saturation=(report['adapter_saturation_samples'] / report['samples']
                                   - reference['adapter_saturation_samples'] / reference['samples']) <= .05,
    )
