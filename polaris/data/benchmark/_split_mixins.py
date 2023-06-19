from typing import Sequence
from pydantic import BaseModel, root_validator


class SingleTaskSplitMixin(BaseModel):
    """Mixin for any single-task benchmark specification"""

    @root_validator(skip_on_failure=True)
    def validate_single_task_split(cls, values):
        """
        A valid single task split assigns inputs exclusively to a single partition.
        It is not required that the split covers the entirety of a dataset.
        """
        v = values["split"]
        train_indices = v[0]
        test_indices = [i for part in v[1].values() for i in part] if isinstance(v[1], dict) else v[1]
        if any(i < 0 or i >= len(values["dataset"]) for i in train_indices + test_indices):
            raise ValueError("The predefined split contains invalid indices")
        if any(i in train_indices for i in test_indices):
            raise ValueError("The predefined split specifies overlapping train and test sets")
        return values


class MultiTaskSplitMixin(BaseModel):
    """Mixin for any multi-task benchmark specification"""

    @staticmethod
    def check_split_partition(indices, no_datapoints: int, no_targets: int):
        """Helper method to easily check indices for a single partition."""
        checked_indices = []
        for tup in indices:
            if isinstance(tup, Sequence):
                invalid = (
                    len(tup) != 2
                    or tup[0] < 0
                    or tup[0] >= no_datapoints
                    or any(i < 0 for i in tup[1])
                    or any(i >= no_targets for i in tup[1])
                )
            elif isinstance(tup, int):
                # With single index, we assume all targets are indexed.
                # This simplifies split definitions, because you can specify X instead of (X, 1), (X, 2), ...
                # Changing the index here to a consistent format for downstream processing.
                invalid = tup < 0 or tup >= no_datapoints
                tup = (tup, list(range(no_targets)))

            else:
                invalid = True

            if invalid:
                raise ValueError("The predefined split contains invalid indices")

            checked_indices.append(tup)

        return checked_indices

    @root_validator(skip_on_failure=True)
    def validate_multi_task_split(cls, values):
        """
        A valid multitask split assigns each input, target pair exclusively to a single partition.
        It is not required that the split covers the entirety of a dataset.
        """

        # Extract the relevant data
        v = values["split"]
        no_datapoints = len(values["dataset"])
        no_targets = len(values["target_cols"])

        # Check the train set
        train_pairs = cls.check_split_partition(v[0], no_datapoints, no_targets)

        # Check the test sets and whether any of them overlap with train
        if isinstance(v[1], dict):
            test_pairs = {k: cls.check_split_partition(v, no_datapoints, no_targets) for k, v in v[1].items()}
            overlapping = any(tup in train_pairs for subset in test_pairs.values() for tup in subset)
        else:
            test_pairs = cls.check_split_partition(v[1], no_datapoints, no_targets)
            overlapping = any(tup in train_pairs for tup in test_pairs)

        if overlapping:
            raise ValueError("The predefined split specifies overlapping train and test sets")

        values["split"] = train_pairs, test_pairs
        return values
