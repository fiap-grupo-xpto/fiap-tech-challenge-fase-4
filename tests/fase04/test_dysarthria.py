from backend.rehab_audio.dysarthria import DEFAULT_LIMITS, dysarthria_findings


def test_no_flag_when_markers_within_norm():
    metrics = {"jitter_local": 0.5, "shimmer_local": 2.0, "hnr": 25.0}

    assert dysarthria_findings(metrics) == []


def test_single_deviant_marker_is_not_enough():
    metrics = {"jitter_local": 2.0, "shimmer_local": 2.0, "hnr": 25.0}

    assert dysarthria_findings(metrics) == []


def test_two_deviant_markers_raise_possible_dysarthria():
    metrics = {"jitter_local": 2.0, "shimmer_local": 5.0, "hnr": 12.0}

    findings = dysarthria_findings(metrics)

    assert [item["code"] for item in findings] == ["possivel_disartria"]
    assert findings[0]["severity"] == "warning"


def test_missing_or_zero_markers_are_ignored():
    assert dysarthria_findings({}) == []
    assert (
        dysarthria_findings({"jitter_local": 0.0, "shimmer_local": 0.0, "hnr": 0.0})
        == []
    )


def test_default_limits_expose_normophonia_cut_offs():
    assert set(DEFAULT_LIMITS) == {"jitter_local", "shimmer_local", "hnr"}


def test_injected_limits_change_the_decision():
    metrics = {"jitter_local": 0.5, "shimmer_local": 2.0, "hnr": 25.0}

    assert dysarthria_findings(metrics) == []

    strict = {"jitter_local": 0.4, "shimmer_local": 1.8, "hnr": 30.0}

    findings = dysarthria_findings(metrics, strict)

    assert [item["code"] for item in findings] == ["possivel_disartria"]
    assert findings[0]["expected"] in strict.values()
