def test_all_dependencies_are_pinned():
    with open("docker/requirements.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    unconstrained = [l for l in lines if not any(op in l for op in ("==", ">=", "<=", "~="))]
    assert not unconstrained, f"dependencies with no version constraint: {unconstrained}"