from typing import List, TypeVar

T = TypeVar("T")


class RiskThreshold:
    def __init__(self, thresholds: List[float], labels: List[T]):
        assert len(labels) == len(thresholds) + 1, "Number of labels must be one more than number of thresholds"
        assert all(
            [t1 < t2 for t1, t2 in zip(thresholds[:-1], thresholds[1:])]
        ), "Thresholds must be in increasing order"
        assert thresholds[0] > 0, "Thresholds must be greater than than 0"
        assert thresholds[-1] < 1, "Thresholds must be smaller than 1"
        assert len(set(labels)) == len(labels), "Labels must be all different"
        # Generate range-based thresholds
        self._thresholds = (
            [(0, thresholds[0], labels[0])]
            + [(t1, t2, l) for t1, t2, l in zip(thresholds[:-1], thresholds[1:], labels[1:])]
            + [(thresholds[-1], 1.0 + 1e-6, labels[-1])]
        )
        # 1+1e-6 to avoid missing 1.0 as highest threshold

    @property
    def highest(self) -> T:
        return self._thresholds[-1][2]

    @property
    def thresholds(self) -> List[float]:
        return [t[1] for t in self._thresholds[:-1]]

    @property
    def labels(self) -> List[T]:
        return [t[2] for t in self._thresholds]

    def membership(self, value: float) -> T:
        assert value >= 0, "Value must be greater than or equal to 0"
        assert value <= 1, "Value must be less than or equal to 1"
        for l, h, label in self._thresholds:
            if value >= l and value < h:
                return label

    def range_membership(self, low: float, high: float) -> List[T]:
        assert low <= high, "Low must be less than or equal to high"
        assert low >= 0, "Low must be greater than or equal to 0"
        assert high <= 1, "High must be less than or equal to 1"
        labels = []
        for l, h, label in self._thresholds:
            # If the low value is within the range defined by the threshold, add the label
            if low >= l and low < h and label not in labels:
                labels.append(label)
            # If the high value is within the range defined by the threshold, add the label
            if high > l and high <= h and label not in labels:
                labels.append(label)
            # If the low and high values span multiple thresholds, add the label for each threshold
            if low < l and high > h and label not in labels:
                labels.append(label)
        return labels
