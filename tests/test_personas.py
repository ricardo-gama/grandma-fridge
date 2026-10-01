from src.grandma_personas import PERSONAS, REFUSAL

EXPECTED_PERSONAS = {"affectionate", "strict", "dramatic", "practical"}


def test_all_four_personas_present():
    assert set(PERSONAS.keys()) == EXPECTED_PERSONAS


def test_every_template_has_fridge_items_placeholder():
    for name, persona in PERSONAS.items():
        assert "{{fridge_items}}" in persona["template"], f"{name} is missing the fridge_items placeholder"


def test_every_template_gates_refusal_explicitly():
    # Regression guard: a template that only says "if asked about anything else, refuse"
    # has no rule for judging whether fridge_items itself is real food — that's the bug
    # that made 'affectionate' write a recipe for cement, screws and a screwdriver.
    for name, persona in PERSONAS.items():
        template_lower = persona["template"].lower()
        assert "rule:" in template_lower, f"{name} has no explicit food/non-food gate"
        assert REFUSAL in persona["template"], f"{name} doesn't reference the shared refusal text"