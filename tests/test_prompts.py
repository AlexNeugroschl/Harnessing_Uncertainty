from empg.prompts import alfworld_prompt, parse_action


def test_parse_valid():
    action, ok = parse_action("<think>mug is on desk</think><action>Go to desk 1</action>")
    assert ok and action == "go to desk 1"


def test_parse_missing_think_is_invalid():
    action, ok = parse_action("<action>go to desk 1</action>")
    assert not ok and action == "go to desk 1"


def test_parse_no_action_is_invalid():
    _, ok = parse_action("<think>hmm</think> I will go to the desk")
    assert not ok


def test_parse_two_actions_is_invalid():
    _, ok = parse_action("<think>x</think><action>look</action><action>inventory</action>")
    assert not ok


def test_prompt_first_step_and_history():
    p0 = alfworld_prompt("put a mug in desk", "You are in a room.", ["look", "help"], [])
    assert "'look'" in p0 and "'help'" not in p0 and "Your task is to" not in p0
    hist = [("o1", "a1"), ("o2", "a2"), ("o3", "a3")]
    p = alfworld_prompt("put a mug in desk", "o4", ["look"], hist, history_length=2)
    assert "Your task is to: put a mug in desk" in p
    assert "Observation 2: 'o2'" in p and "Observation 1" not in p
    assert "step 4" in p
