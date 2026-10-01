import yaml

COMPOSE_PATH = "docker/docker-compose.yml"


def load_compose():
    with open(COMPOSE_PATH) as f:
        return yaml.safe_load(f)


def test_required_services_exist():
    services = load_compose()["services"]
    for name in ("mlflow", "jupyter", "api"):
        assert name in services, f"missing service: {name}"


def test_api_depends_on_mlflow():
    services = load_compose()["services"]
    assert "mlflow" in services["api"].get("depends_on", [])


def test_api_and_jupyter_load_secrets_from_env_file_not_inline():
    services = load_compose()["services"]
    for name in ("api", "jupyter"):
        svc = services[name]
        assert "env_file" in svc, f"{name} should load secrets via env_file"
        inline_env = svc.get("environment", {})
        blob = yaml.dump(inline_env).lower()
        assert "key" not in blob, f"{name} must not hardcode a key inline — found one in `environment:`"


def test_mlflow_port_avoids_macos_airplay_conflict():
    # Port 5000 clashes with AirPlay Receiver on macOS — the README documents this.
    ports = load_compose()["services"]["mlflow"]["ports"]
    assert "5000:5000" not in ports