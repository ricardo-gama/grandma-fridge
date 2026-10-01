DOCKERFILES = ["docker/Dockerfile.api", "docker/Dockerfile.jupyter"]


def test_python_version_pinned():
    for path in DOCKERFILES:
        text = open(path).read()
        assert "3.12" in text, f"{path} should pin Python 3.12 (see README troubleshooting: 3.13 breaks pyarrow/pkg_resources wheels)"


def test_requirements_copied_before_app_code():
    # Keeps Docker's layer cache useful — app-code edits shouldn't force a pip reinstall.
    text = open("docker/Dockerfile.api").read()
    req_line = text.find("requirements.txt")
    copy_src = text.find("COPY api")
    assert 0 < req_line < copy_src