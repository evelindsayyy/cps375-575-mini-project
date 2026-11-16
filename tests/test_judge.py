from src.judge import judge_messages, parse_verdict


def test_parse_verdict_reads_only_a_final_verdict_line():
    assert parse_verdict("- Use lowercase: FOLLOWED\nVERDICT: PASS") == "pass"
    assert parse_verdict("- Under 75 words: NOT FOLLOWED (82 words)\nVERDICT: FAIL\n") == "fail"
    assert parse_verdict("VERDICT: PASS\nActually, one more thing.") is None
    assert parse_verdict("No verdict at all.") is None


def test_judge_sees_the_request_and_the_response():
    messages = judge_messages("Write two lines.", "Line one.\nLine two.")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Write two lines." in messages[1]["content"] and "Line one.\nLine two." in messages[1]["content"]


def test_judge_is_given_the_measured_word_count():
    content = judge_messages("Use under 10 words.", "One two, three.\n\nFour five.")[1]["content"]
    assert "words in the response: 5" in content and "paragraph" not in content.split("MEASURED")[1]
