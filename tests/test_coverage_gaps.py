import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.aiops_pipeline import load_data, run_pipeline
from src.anomaly_detector import AnomalyDetector
from src.calculations import area_of_circle, get_nth_fibonacci
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_area_of_circle_negative_radius_raises():
    with pytest.raises(ValueError, match="Radius cannot be negative"):
        area_of_circle(-1)


def test_get_nth_fibonacci_negative_raises():
    with pytest.raises(ValueError, match="n cannot be negative"):
        get_nth_fibonacci(-1)


def test_get_nth_fibonacci_iterative_branch():
    assert get_nth_fibonacci(10) == 55


def test_anomaly_detector_detects_cpu_memory_and_warning():
    detector = AnomalyDetector()
    record = {
        "timestamp": "2026-09-20T10:10:00",
        "service": "payment-service",
        "response_time_ms": 100,
        "cpu_percent": 95,
        "memory_percent": 96,
        "log_level": "WARNING",
        "message": "Resource usage elevated",
    }

    event = detector.detect(record)

    assert event is not None
    assert "High CPU utilization" in event["reasons"]
    assert "High memory utilization" in event["reasons"]
    assert "Error log detected" in event["reasons"]


def test_event_producer_returns_false_for_empty_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)

    assert producer.publish(None) is False
    assert topic.get_messages() == []


def test_event_topic_clear_removes_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})
    topic.clear()

    assert topic.get_messages() == []


def test_load_data_reads_json_file(tmp_path):
    data = [{"service": "checkout"}]
    file_path = tmp_path / "records.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    assert load_data(str(file_path)) == data


def test_run_pipeline_returns_detected_events_and_empty_consumed(tmp_path):
    data = [
        {
            "timestamp": "2026-09-20T10:00:00",
            "service": "payment-service",
            "response_time_ms": 100,
            "cpu_percent": 10,
            "memory_percent": 10,
            "log_level": "INFO",
            "message": "ok",
        },
        {
            "timestamp": "2026-09-20T10:01:00",
            "service": "payment-service",
            "response_time_ms": 700,
            "cpu_percent": 90,
            "memory_percent": 90,
            "log_level": "ERROR",
            "message": "slow",
        },
    ]
    file_path = tmp_path / "service_data.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    result = run_pipeline(str(file_path))

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_main_entrypoint_prints_consumed_events(monkeypatch, capsys):
    import event_consumer

    monkeypatch.chdir(ROOT_DIR)
    monkeypatch.setattr(
        event_consumer.EventConsumer,
        "consume",
        lambda self: [
            {
                "service": "payment-service",
                "timestamp": "2026-09-20T10:05:00",
                "type": "ANOMALY",
                "reasons": ["High response time"],
            }
        ],
    )

    runpy.run_path(str(SRC_DIR / "aiops_pipeline.py"), run_name="__main__")
    output = capsys.readouterr().out

    assert "AIOps Pipeline Result" in output
    assert "Detected Events:" in output
    assert "Service: payment-service" in output
