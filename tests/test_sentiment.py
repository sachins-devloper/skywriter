"""Tests for expression scoring. Pure functions -- no camera, no MediaPipe."""

from airwrite.face import NEUTRAL_THRESHOLD, classify, score_expressions


def blend(**kwargs):
    return dict(kwargs)


class TestClassify:
    def test_smile_reads_as_happy(self):
        label, conf = classify(blend(mouthSmileLeft=0.9, mouthSmileRight=0.9))
        assert label == "happy"
        assert conf > NEUTRAL_THRESHOLD

    def test_frown_reads_as_sad(self):
        label, _ = classify(blend(mouthFrownLeft=0.8, mouthFrownRight=0.8,
                                  browInnerUp=0.5))
        assert label == "sad"

    def test_open_jaw_and_wide_eyes_read_as_surprised(self):
        label, _ = classify(blend(jawOpen=0.9, eyeWideLeft=0.8,
                                  eyeWideRight=0.8, browOuterUpLeft=0.6,
                                  browOuterUpRight=0.6))
        assert label == "surprised"

    def test_lowered_brows_read_as_angry(self):
        label, _ = classify(blend(browDownLeft=0.9, browDownRight=0.9,
                                  noseSneerLeft=0.5, noseSneerRight=0.5))
        assert label == "angry"

    def test_slack_face_reads_as_neutral(self):
        label, _ = classify(blend(mouthSmileLeft=0.03, jawOpen=0.02))
        assert label == "neutral"

    def test_no_face_reads_as_neutral(self):
        assert classify({}) == ("neutral", 0.0)

    def test_unknown_blendshapes_are_ignored(self):
        # A model revision adding new shapes must not crash the scorer.
        label, _ = classify(blend(mouthSmileLeft=0.9, mouthSmileRight=0.9,
                                  somethingNew=0.99))
        assert label == "happy"


class TestScores:
    def test_every_expression_is_scored(self):
        scores = score_expressions(blend(mouthSmileLeft=0.5))
        assert set(scores) == {"happy", "sad", "surprised", "angry"}

    def test_scores_are_normalised_to_unit_range(self):
        # All contributing shapes maxed must not exceed 1.0, or the confidence
        # percentage shown on screen would read above 100%.
        maxed = blend(mouthSmileLeft=1.0, mouthSmileRight=1.0,
                      cheekSquintLeft=1.0, cheekSquintRight=1.0)
        assert score_expressions(maxed)["happy"] <= 1.0

    def test_stronger_expression_scores_higher(self):
        weak = score_expressions(blend(mouthSmileLeft=0.2, mouthSmileRight=0.2))
        strong = score_expressions(blend(mouthSmileLeft=0.9, mouthSmileRight=0.9))
        assert strong["happy"] > weak["happy"]

    def test_expressions_are_distinguishable(self):
        smile = score_expressions(blend(mouthSmileLeft=0.9, mouthSmileRight=0.9))
        assert smile["happy"] > smile["sad"]
        assert smile["happy"] > smile["angry"]
