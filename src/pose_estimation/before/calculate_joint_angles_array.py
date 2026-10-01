"""Calculate COCO 3-D joint Euler angles from in-memory NumPy arrays.

Input format:
    keypoints.shape == (frames, 17, 3)

The 17 points must follow the standard COCO order exposed by
:const:`COCO_JOINTS`.  The return value is a numeric array with shape
``(frames, 15, 3)``.  Its last axis is ``[Roll, Pitch, Yaw]`` in degrees and
its middle axis follows :const:`OUTPUT_JOINTS`.
"""

import numpy as np


COCO_JOINTS = np.array(
    [
        "nose",
        "lefteye",
        "righteye",
        "leftear",
        "rightear",
        "leftshoulder",
        "rightshoulder",
        "leftelbow",
        "rightelbow",
        "leftwrist",
        "rightwrist",
        "lefthip",
        "righthip",
        "leftknee",
        "rightknee",
        "leftankle",
        "rightankle",
    ]
)

OUTPUT_JOINTS = np.array(
    [
        "hips",
        "neck",
        "head",
        "L_shoulder",
        "L_elbow",
        "L_wrist",
        "R_shoulder",
        "R_elbow",
        "R_wrist",
        "L_hip",
        "L_knee",
        "L_ankle",
        "R_hip",
        "R_knee",
        "R_ankle",
    ]
)

# Parent relationship for the output joint axis.
PARENTS = {
    "hips": None,
    "neck": "hips",
    "head": "neck",
    "L_shoulder": "neck",
    "L_elbow": "L_shoulder",
    "L_wrist": "L_elbow",
    "R_shoulder": "neck",
    "R_elbow": "R_shoulder",
    "R_wrist": "R_elbow",
    "L_hip": "hips",
    "L_knee": "L_hip",
    "L_ankle": "L_knee",
    "R_hip": "hips",
    "R_knee": "R_hip",
    "R_ankle": "R_knee",
}

# Each joint frame uses its outgoing bone as the local X direction.
_CHILDREN = {
    "neck": "head",
    "head": None,
    "L_shoulder": "L_elbow",
    "L_elbow": "L_wrist",
    "L_wrist": None,
    "R_shoulder": "R_elbow",
    "R_elbow": "R_wrist",
    "R_wrist": None,
    "L_hip": "L_knee",
    "L_knee": "L_ankle",
    "L_ankle": None,
    "R_hip": "R_knee",
    "R_knee": "R_ankle",
    "R_ankle": None,
}


def _unit(vector, label):
    """Return a unit vector, raising a useful error for a zero-length bone."""
    length = np.linalg.norm(vector)
    if length <= np.finfo(float).eps:
        raise ValueError(f"Cannot calculate orientation: zero-length vector for {label}")
    return vector / length


def _frame_from_direction(direction, reference, label):
    """Build an orthonormal frame whose first axis follows ``direction``."""
    x_axis = _unit(direction, label)
    reference = np.asarray(reference, dtype=float)
    y_axis = reference - np.dot(reference, x_axis) * x_axis

    # A bone parallel to the reference needs a different reference axis.
    if np.linalg.norm(y_axis) <= np.finfo(float).eps:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, x_axis)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        y_axis = fallback - np.dot(fallback, x_axis) * x_axis

    y_axis = _unit(y_axis, label + " reference")
    z_axis = _unit(np.cross(x_axis, y_axis), label + " cross product")
    y_axis = _unit(np.cross(z_axis, x_axis), label + " orthogonal axis")
    return np.column_stack((x_axis, y_axis, z_axis))


def _body_frame(points):
    """Return the body frame: X right, Y upward, Z from their cross product."""
    right = points["rightshoulder"] - points["leftshoulder"]
    up = points["neck"] - points["hips"]
    x_axis = _unit(right, "shoulder line")
    y_axis = up - np.dot(up, x_axis) * x_axis
    y_axis = _unit(y_axis, "body vertical")
    z_axis = _unit(np.cross(x_axis, y_axis), "body cross product")
    y_axis = _unit(np.cross(z_axis, x_axis), "body vertical")
    return np.column_stack((x_axis, y_axis, z_axis))


def _head_frame(points, body_frame):
    """Build a head frame from the five facial points."""
    right = points["rightear"] - points["leftear"]
    forward = points["nose"] - points["head"]
    return _frame_from_direction(right, forward, "head")


def _named_points(frame):
    """Map one COCO frame to the named points used by the skeleton."""
    points = {name: frame[index] for index, name in enumerate(COCO_JOINTS)}
    points["hips"] = (points["lefthip"] + points["righthip"]) / 2.0
    points["neck"] = (points["leftshoulder"] + points["rightshoulder"]) / 2.0
    face = np.array(
        [
            points["nose"],
            points["lefteye"],
            points["righteye"],
            points["leftear"],
            points["rightear"],
        ]
    )
    points["head"] = np.mean(face, axis=0)
    points["leftfoot"] = points["leftankle"]
    points["rightfoot"] = points["rightankle"]
    points["L_shoulder"] = points["leftshoulder"]
    points["L_elbow"] = points["leftelbow"]
    points["L_wrist"] = points["leftwrist"]
    points["R_shoulder"] = points["rightshoulder"]
    points["R_elbow"] = points["rightelbow"]
    points["R_wrist"] = points["rightwrist"]
    points["L_hip"] = points["lefthip"]
    points["L_knee"] = points["leftknee"]
    points["L_ankle"] = points["leftankle"]
    points["R_hip"] = points["righthip"]
    points["R_knee"] = points["rightknee"]
    points["R_ankle"] = points["rightankle"]
    return points


def _euler_zxy(rotation):
    """Decompose Rz @ Rx @ Ry and return [Roll, Pitch, Yaw] in degrees."""
    # This is the same Z-X-Y decomposition used by the original utils.py.
    roll = np.arctan2(-rotation[0, 1], rotation[1, 1])
    pitch = np.arctan2(
        rotation[2, 1],
        np.sqrt(rotation[2, 0] ** 2 + rotation[2, 2] ** 2),
    )
    yaw = np.arctan2(-rotation[2, 0], rotation[2, 2])
    return np.degrees([roll, pitch, yaw])


def _joint_frames(points):
    body_frame = _body_frame(points)
    frames = {"hips": body_frame}

    for joint in OUTPUT_JOINTS:
        if joint == "hips":
            continue
        parent = PARENTS[joint]
        child = _CHILDREN[joint]

        if joint == "head":
            frames[joint] = _head_frame(points, body_frame)
            continue

        if child is None:
            direction = points[joint] - points[parent]
        else:
            direction = points[child] - points[joint]
        frames[joint] = _frame_from_direction(
            direction,
            body_frame[:, 1],
            joint,
        )

    return frames


def calculate_joint_angles(keypoints: np.ndarray) -> np.ndarray:
    """Calculate all requested joint angles from COCO 17-point coordinates.

    Args:
        keypoints: Floating-point coordinates with shape ``(frames, 17, 3)``.
            The input is interpreted in standard COCO order and is never
            modified.

    Returns:
        A floating-point NumPy array with shape ``(frames, 15, 3)``.  The
        middle axis follows ``OUTPUT_JOINTS`` and the last axis is
        ``[Roll, Pitch, Yaw]`` in degrees.  Euler angles use the existing
        ``Rz @ Rx @ Ry`` (Z-X-Y) convention.

    Raises:
        TypeError: If ``keypoints`` is not a NumPy array.
        ValueError: If the shape, values, or geometry are invalid.

    Notes:
        A wrist or ankle has no observable twist around its incoming bone when
        only joint centers are available. Its frame therefore uses the body
        vertical direction to deterministically fix that otherwise ambiguous
        axis.
    """
    if not isinstance(keypoints, np.ndarray):
        raise TypeError("keypoints must be a NumPy array")
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError("keypoints must have shape (frames, 17, 3)")
    if keypoints.shape[0] == 0:
        raise ValueError("keypoints must contain at least one frame")
    if not np.issubdtype(keypoints.dtype, np.number):
        raise TypeError("keypoints must contain numeric values")
    if not np.all(np.isfinite(keypoints)):
        raise ValueError("keypoints must contain only finite values")

    angles = np.empty((keypoints.shape[0], len(OUTPUT_JOINTS), 3), dtype=float)
    for frame_index, frame in enumerate(keypoints):
        points = _named_points(frame.astype(float, copy=False))
        frames = _joint_frames(points)
        for joint_index, joint in enumerate(OUTPUT_JOINTS):
            parent = PARENTS[joint]
            rotation = frames[joint] if parent is None else frames[parent].T @ frames[joint]
            angles[frame_index, joint_index] = _euler_zxy(rotation)

    return angles


__all__ = ["COCO_JOINTS", "OUTPUT_JOINTS", "calculate_joint_angles"]
