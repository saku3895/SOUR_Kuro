"""Fixed upper-body skeleton definition for motion-language validation."""

import numpy as np


JOINT_NAMES = (
    "hips",
    "neck",
    "L_shoulder",
    "L_elbow",
    "R_shoulder",
    "R_elbow",
)
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

# Relative vectors in each parent's local frame. Units are arbitrary and
# describe a stable validation model, not a measured person's proportions.
BONE_VECTORS = {
    ("hips", "neck"): np.array([0.0, 0.30, 0.0]),
    ("neck", "L_shoulder"): np.array([-0.20, 0.0, 0.0]),
    ("L_shoulder", "L_elbow"): np.array([-0.30, 0.0, 0.0]),
    ("neck", "R_shoulder"): np.array([0.20, 0.0, 0.0]),
    ("R_shoulder", "R_elbow"): np.array([0.30, 0.0, 0.0]),
}

EDGES = tuple(
    (JOINT_INDEX[parent], JOINT_INDEX[child])
    for parent, child in BONE_VECTORS
)

PARENTS = {
    "hips": None,
    "neck": "hips",
    "L_shoulder": "neck",
    "L_elbow": "L_shoulder",
    "R_shoulder": "neck",
    "R_elbow": "R_shoulder",
}
