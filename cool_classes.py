"""Ordered MPEG7 classes preferred for sampling."""

import operator


COOL_CLASSES = (
    (47, "hat"),
    (40, "flatfish"),
    (4, "Heart"),
    (2, "Glas"),
    (0, "Bone"),
    (7, "bat"),
    (12, "brick"),
    (31, "device5"),
    (34, "device8"),
    (35, "device9"),
    (61, "shoe"),
    (11, "bottle"),
    (1, "Comma"),
    (49, "horseshoe"),
    (60, "sea_snake"),
    (9, "bell"),
    (62, "spoon"),
    (29, "device3"),
    (30, "device4"),
    (26, "device0"),
    (27, "device1"),
    (68, "turtle"),
    (24, "cup"),
    (45, "guitar"),
    (50, "jar"),
    (51, "key"),
    (56, "personal_car"),
    (59, "ray"),
    (66, "tree"),
    (46, "hammer"),
)
COOL_CLASS_IDS = tuple(class_id for class_id, _ in COOL_CLASSES)
COOL_CLASS_NAMES = tuple(class_name for _, class_name in COOL_CLASSES)
NUM_MPEG7_CLASSES = 70


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not {value!r}")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer, not {value!r}") from error


def resolve_cool_classes(num_cool_classes=None, cool_class_ids=None):
    """Resolve count/current-list or explicit/restored IDs to sampler metadata."""
    if cool_class_ids is None:
        if num_cool_classes is None:
            return None, None
        count = _integer(num_cool_classes, "num_cool_classes")
        if not 1 <= count <= len(COOL_CLASS_IDS):
            raise ValueError(
                f"num_cool_classes must be between 1 and {len(COOL_CLASS_IDS)}, "
                f"got {count}"
            )
        return count, COOL_CLASS_IDS[:count]

    try:
        class_ids = tuple(
            _integer(class_id, "cool_class_ids entries")
            for class_id in cool_class_ids
        )
    except TypeError as error:
        raise ValueError("cool_class_ids must be an iterable of integers") from error

    if not class_ids:
        raise ValueError("cool_class_ids must not be empty")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("cool_class_ids must not contain duplicates")
    invalid = [class_id for class_id in class_ids if not 0 <= class_id < NUM_MPEG7_CLASSES]
    if invalid:
        raise ValueError(
            f"cool_class_ids must be between 0 and {NUM_MPEG7_CLASSES - 1}, "
            f"got {invalid}"
        )

    if num_cool_classes is None:
        count = len(class_ids)
    else:
        count = _integer(num_cool_classes, "num_cool_classes")
        if count != len(class_ids):
            raise ValueError(
                f"num_cool_classes ({count}) must match the number of "
                f"cool_class_ids ({len(class_ids)})"
            )
    return count, class_ids


def validate_cool_classes(class_names):
    """Ensure the preference IDs still refer to the expected MPEG7 names."""
    if len(class_names) != NUM_MPEG7_CLASSES:
        raise ValueError(
            f"Expected {NUM_MPEG7_CLASSES} MPEG7 classes, got {len(class_names)}"
        )
    mismatches = [
        (class_id, class_name, class_names[class_id])
        for class_id, class_name in COOL_CLASSES
        if class_names[class_id] != class_name
    ]
    if mismatches:
        raise ValueError(f"Cool class IDs do not match MPEG7 class names: {mismatches}")
