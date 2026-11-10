"""Never deliver test alerts to a configured real channel or production database."""
import pytest


@pytest.fixture(autouse=True)
def isolate_notification_delivery(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "")
    # Upload-cleanup tests own tmp_path and assert it is empty afterwards.
    database_dir = tmp_path_factory.mktemp("notification-db")
    monkeypatch.setenv("ANOMALY_DB_PATH", str(database_dir / "notifications.db"))
