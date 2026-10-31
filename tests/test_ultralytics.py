"""Tests for the Ultralytics helpers, using stand-ins for the result objects."""

import pytest

import trackforge


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


class FakeBoxes:
    def __init__(self, rows):
        self.xyxy = FakeTensor([r[:4] for r in rows])
        self.conf = FakeTensor([r[4] for r in rows])
        self.cls = FakeTensor([float(r[5]) for r in rows])
        self._n = len(rows)

    def __len__(self):
        return self._n


class FakeResult:
    def __init__(self, rows, frame="frame"):
        self.boxes = FakeBoxes(rows) if rows is not None else None
        self.orig_img = frame


class FakeModel:
    def __init__(self, frames):
        self.frames = frames
        self.kwargs = None

    def predict(self, source, **kwargs):
        self.kwargs = kwargs
        return iter(self.frames)


ROWS = [
    [10.0, 20.0, 50.0, 80.0, 0.9, 0],
    [100.0, 100.0, 160.0, 220.0, 0.75, 2],
]


def test_from_ultralytics_converts_xyxy_to_tlwh():
    dets = trackforge.from_ultralytics(FakeResult(ROWS))
    assert dets == [
        ([10.0, 20.0, 40.0, 60.0], 0.9, 0),
        ([100.0, 100.0, 60.0, 120.0], 0.75, 2),
    ]
    assert all(isinstance(d[2], int) for d in dets)


def test_from_ultralytics_accepts_single_result_list():
    assert trackforge.from_ultralytics(
        [FakeResult(ROWS)]
    ) == trackforge.from_ultralytics(FakeResult(ROWS))


def test_from_ultralytics_empty_and_missing_boxes():
    assert trackforge.from_ultralytics(FakeResult([])) == []
    assert trackforge.from_ultralytics(FakeResult(None)) == []


def test_from_ultralytics_rejects_batches():
    with pytest.raises(ValueError):
        trackforge.from_ultralytics([FakeResult(ROWS), FakeResult(ROWS)])


def test_track_ultralytics_keeps_ids_across_frames():
    moved = [[x + 5.0 if i < 4 else x for i, x in enumerate(r)] for r in ROWS]
    model = FakeModel([FakeResult(ROWS), FakeResult(moved)])
    tracker = trackforge.BYTETRACK()

    frames = list(
        trackforge.track_ultralytics(model, "video.mp4", tracker, classes=[0, 2])
    )

    assert len(frames) == 2
    assert model.kwargs == {"classes": [0, 2], "stream": True, "verbose": False}
    first_ids = sorted(t[0] for t in frames[0][1])
    second_ids = sorted(t[0] for t in frames[1][1])
    assert first_ids == second_ids == [1, 2]


def test_track_ultralytics_passes_embeddings():
    seen = []

    def embedder(frame, boxes):
        seen.append((frame, boxes))
        return [[1.0, 0.0] for _ in boxes]

    model = FakeModel([FakeResult(ROWS, frame="f0")])
    tracker = trackforge.DEEPSORT()

    list(trackforge.track_ultralytics(model, 0, tracker, embedder=embedder))

    assert seen == [("f0", [[10.0, 20.0, 40.0, 60.0], [100.0, 100.0, 60.0, 120.0]])]
