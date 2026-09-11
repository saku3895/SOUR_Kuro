"""Detect motion intervals from 15-joint Euler-angle frames."""

from collections import deque

import numpy as np


class MotionTriggerEngine:
    """Track angle motion and return completed ``(frames, 15, 3)`` sequences."""

    IDLE = 0
    TRACKING = 1

    def __init__(
        self,
        threshold_energy=4.0,
        release_threshold=2.0,
        min_trigger_frames=6,
        stability_frames=8,
        smoothing_frames=10,
    ):
        if threshold_energy <= 0 or release_threshold < 0:
            raise ValueError("energy thresholds must be non-negative")
        if release_threshold >= threshold_energy:
            raise ValueError("release_threshold must be lower than threshold_energy")
        if min_trigger_frames < 1 or stability_frames < 1 or smoothing_frames < 1:
            raise ValueError("frame counts must be positive")

        self.threshold_energy = float(threshold_energy)
        self.release_threshold = float(release_threshold)
        self.min_trigger_frames = int(min_trigger_frames)
        self.stability_frames = int(stability_frames)
        self.current_state = self.IDLE
        self.gesture_buffer = []
        self.above_threshold_counter = 0
        self.below_threshold_counter = 0
        self.previous_angles = None
        self.energy_history = deque(maxlen=smoothing_frames)
        self.last_raw_energy = 0.0
        self.last_smoothed_energy = 0.0

    @staticmethod
    def _angle_difference(current_angles, previous_angles):
        difference = current_angles - previous_angles
        return (difference + 180.0) % 360.0 - 180.0

    def process_frame(self, current_angles):
        """Process one ``(15, 3)`` Euler-angle frame.

        Returns:
            A tuple ``(smoothed_energy, extracted_sequence)``. The second
            value is ``None`` except on TRIGGER OFF, when it has shape
            ``(frames, 15, 3)``.
        """
        angles = np.asarray(current_angles, dtype=float)
        if angles.shape != (15, 3):
            raise ValueError("current_angles must have shape (15, 3)")
        if not np.all(np.isfinite(angles)):
            raise ValueError("current_angles must contain only finite values")

        if self.previous_angles is None:
            self.previous_angles = angles.copy()
            self.energy_history.append(0.0)
            self.last_raw_energy = 0.0
            self.last_smoothed_energy = 0.0
            return 0.0, None

        difference = self._angle_difference(angles, self.previous_angles)
        raw_energy = float(np.mean(difference ** 2))
        self.previous_angles = angles.copy()
        self.energy_history.append(raw_energy)
        smoothed_energy = float(np.mean(self.energy_history))
        self.last_raw_energy = raw_energy
        self.last_smoothed_energy = smoothed_energy
        extracted_sequence = None

        if self.current_state == self.IDLE:
            if smoothed_energy >= self.threshold_energy:
                self.above_threshold_counter += 1
                if self.above_threshold_counter >= self.min_trigger_frames:
                    self.current_state = self.TRACKING
                    self.gesture_buffer = [angles.copy()]
                    self.below_threshold_counter = 0
                    print("\n[TRIGGER ON] 角度記録を開始")
            else:
                self.above_threshold_counter = 0
        else:
            self.gesture_buffer.append(angles.copy())
            if smoothed_energy <= self.release_threshold:
                self.below_threshold_counter += 1
                if self.below_threshold_counter >= self.stability_frames:
                    extracted_sequence = np.asarray(self.gesture_buffer, dtype=float)
                    print("[TRIGGER OFF] 角度記録を終了")
                    self.current_state = self.IDLE
                    self.gesture_buffer = []
                    self.above_threshold_counter = 0
                    self.below_threshold_counter = 0
            else:
                self.below_threshold_counter = 0

        return smoothed_energy, extracted_sequence


__all__ = ["MotionTriggerEngine"]
