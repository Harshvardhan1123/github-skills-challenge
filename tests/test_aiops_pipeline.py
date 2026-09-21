import json

from src.anomaly_detector import AnomalyDetector
from src.aiops_pipeline import load_data, run_pipeline
from src.event_consumer import EventConsumer
from src.event_producer import EventProducer
from src.event_topic import EventTopic


def test_normal_record_is_not_anomaly():
    detector = AnomalyDetector()

    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 120,
        "cpu_percent": 42,
        "memory_percent": 51,
        "log_level": "INFO",
        "message": "Payment request processed successfully"
    }

    assert detector.detect(record) is None


def test_anomalous_record_is_detected():
    detector = AnomalyDetector()

    record = {
        "timestamp": "2026-09-20T10:05:00",
        "service": "payment-service",
        "response_time_ms": 610,
        "cpu_percent": 75,
        "memory_percent": 70,
        "log_level": "ERROR",
        "message": "Payment service timeout"
    }

    event = detector.detect(record)

    assert event is not None
    assert event["type"] == "ANOMALY"


def test_producer_publishes_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)

    event = {
        "type": "ANOMALY",
        "service": "payment-service"
    }

    assert producer.publish(event)
    assert len(topic.get_messages()) == 1


def test_consumer_receives_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)
    consumer = EventConsumer(topic)

    event = {
        "type": "ANOMALY",
        "service": "payment-service"
    }

    producer.publish(event)

    messages = consumer.consume()

    assert len(messages) == 1


def test_load_data_reads_json(tmp_path):
    data = [{"service": "payment-service"}]
    file_path = tmp_path / "service_data.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    assert load_data(str(file_path)) == data


def test_run_pipeline_processes_records(tmp_path):
    file_path = tmp_path / "service_data.json"
    file_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-09-20T10:00:00",
                    "service": "payment-service",
                    "response_time_ms": 120,
                    "cpu_percent": 42,
                    "memory_percent": 51,
                    "log_level": "INFO",
                    "message": "Payment request processed successfully"
                },
                {
                    "timestamp": "2026-09-20T10:05:00",
                    "service": "payment-service",
                    "response_time_ms": 610,
                    "cpu_percent": 95,
                    "memory_percent": 96,
                    "log_level": "WARNING",
                    "message": "Payment service timeout"
                }
            ]
        ),
        encoding="utf-8"
    )

    result = run_pipeline(str(file_path))

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_producer_rejects_empty_event():
    topic = EventTopic("anomaly-events")
    producer = EventProducer(topic)

    assert producer.publish(None) is False


def test_topic_clear_removes_messages():
    topic = EventTopic("anomaly-events")
    topic.publish({"type": "ANOMALY"})

    topic.clear()

    assert topic.get_messages() == []
