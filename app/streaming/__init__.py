"""Streaming ingestion primitives.

Design (mirrors the rest of the codebase):
- A small protocol (`EventSource`) that yields events until exhausted.
- An offline adapter (`InMemoryQueueSource`) for tests and demos.
- An opt-in adapter for a real broker (`KafkaEventSource`, later) behind
  a lazy import and an extras group.

Pull-based (async iterator) rather than push-based because the consumer
controls backpressure by how fast it pulls. This is what Kafka's
consumer API already gives us, so the offline adapter mirrors it.
"""
