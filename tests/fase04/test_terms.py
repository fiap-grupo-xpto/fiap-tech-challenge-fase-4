from backend.rehab_audio.terms import extract_critical_terms, normalize_text


def test_normalize_text_removes_accents_and_lowercases():
    assert normalize_text("Cansaço, TONTURA e DOR!") == "cansaco, tontura e dor!"


def test_extract_critical_terms_is_accent_insensitive():
    terms = extract_critical_terms("Estou com cansaço e muita dor no joelho.")

    found = {(item["term"], item["category"]) for item in terms}
    assert ("cansaco", "sintoma") in found
    assert ("dor", "sintoma") in found


def test_severity_mapping_per_category():
    warning_terms = extract_critical_terms("sinto dor")
    info_terms = extract_critical_terms("repita a serie")

    assert warning_terms[0]["severity"] == "warning"
    assert info_terms[0]["severity"] == "info"


def test_occurrence_counting():
    terms = extract_critical_terms("dor, dor e mais dor")
    dor = next(item for item in terms if item["term"] == "dor")

    assert dor["count"] == 3


def test_empty_input_returns_empty_list():
    assert extract_critical_terms("") == []
    assert extract_critical_terms("   ") == []


def test_common_words_do_not_trigger_false_positives():
    # "para" (preposicao) nao deve casar com termo algum.
    assert extract_critical_terms("vou para casa e preciso dormir agora") == []

    # "dormir" nao deve incrementar a contagem de "dor".
    terms = extract_critical_terms("senti dor nas costas ao dormir de lado")
    dor = next(item for item in terms if item["term"] == "dor")
    assert dor["count"] == 1


def test_escalation_words_are_matched_as_whole_words():
    terms = extract_critical_terms("pare, preciso de ajuda")

    found = {item["term"] for item in terms}
    assert found == {"pare", "ajuda"}


def test_terms_inside_longer_words_are_not_counted():
    assert extract_critical_terms("dorsal e ardor no ombro") == []
