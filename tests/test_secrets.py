def test_env_is_gitignored():
    with open(".gitignore") as f:
        patterns = f.read()
    assert ".env" in patterns


def test_env_example_has_no_real_key():
    with open("docker/.env.example") as f:
        content = f.read()
    assert "GEMINI_API_KEY=" in content
    # the line should be empty after the `=`, not a real-looking key
    key_line = next(l for l in content.splitlines() if l.startswith("GEMINI_API_KEY="))
    assert key_line == "GEMINI_API_KEY="