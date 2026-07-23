//! Prometheus metrics with a deliberately finite, label-free cardinality.

use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::metrics::histogram::{Histogram, exponential_buckets};
use prometheus_client::registry::Registry;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub(crate) struct Metrics {
    registry: Arc<Mutex<Registry>>,
    pub(crate) received: Counter,
    pub(crate) admitted: Counter,
    pub(crate) completed: Counter,
    pub(crate) cancelled: Counter,
    pub(crate) failed: Counter,
    pub(crate) queue_full: Counter,
    pub(crate) idle_prefill_formation: Counter,
    pub(crate) decode_metadata_rebinds: Counter,
    pub(crate) active: Gauge,
    pub(crate) queued: Gauge,
    pub(crate) cache_total_pages: Gauge,
    pub(crate) cache_free_pages: Gauge,
    pub(crate) cache_reserved_pages: Gauge,
    pub(crate) decode_batch_rows: Histogram,
    pub(crate) prefill_batch_rows: Histogram,
    pub(crate) prefill_batch_tokens: Histogram,
    pub(crate) decode_batch_seconds: Histogram,
    pub(crate) prefill_batch_seconds: Histogram,
    pub(crate) ttft_seconds: Histogram,
    pub(crate) tpot_seconds: Histogram,
    pub(crate) request_seconds: Histogram,
    pub(crate) engine_iteration_seconds: Histogram,
}

impl Metrics {
    pub(crate) fn new() -> Self {
        let mut registry = Registry::default();
        let received = Counter::default();
        let admitted = Counter::default();
        let completed = Counter::default();
        let cancelled = Counter::default();
        let failed = Counter::default();
        let queue_full = Counter::default();
        let idle_prefill_formation = Counter::default();
        let decode_metadata_rebinds = Counter::default();
        let active = Gauge::default();
        let queued = Gauge::default();
        let cache_total_pages = Gauge::default();
        let cache_free_pages = Gauge::default();
        let cache_reserved_pages = Gauge::default();
        let decode_batch_rows = Histogram::new(exponential_buckets(1.0, 2.0, 6));
        let prefill_batch_rows = Histogram::new(exponential_buckets(1.0, 2.0, 6));
        let prefill_batch_tokens = Histogram::new(exponential_buckets(16.0, 2.0, 10));
        let decode_batch_seconds = Histogram::new(exponential_buckets(0.000_1, 2.0, 20));
        let prefill_batch_seconds = Histogram::new(exponential_buckets(0.001, 2.0, 20));
        let ttft_seconds = Histogram::new(exponential_buckets(0.001, 2.0, 20));
        let tpot_seconds = Histogram::new(exponential_buckets(0.000_1, 2.0, 20));
        let request_seconds = Histogram::new(exponential_buckets(0.001, 2.0, 24));
        let engine_iteration_seconds = Histogram::new(exponential_buckets(0.000_01, 2.0, 24));
        registry.register(
            "nml_requests_received",
            "Chat completion requests received.",
            received.clone(),
        );
        registry.register(
            "nml_requests_admitted",
            "Requests accepted by the engine command boundary.",
            admitted.clone(),
        );
        registry.register(
            "nml_requests_completed",
            "Requests completed normally.",
            completed.clone(),
        );
        registry.register(
            "nml_requests_cancelled",
            "Requests cancelled before normal completion.",
            cancelled.clone(),
        );
        registry.register(
            "nml_requests_failed",
            "Requests ending in an internal execution failure.",
            failed.clone(),
        );
        registry.register(
            "nml_engine_queue_full",
            "Requests rejected at a bounded engine queue.",
            queue_full.clone(),
        );
        registry.register(
            "nml_engine_idle_prefill_formation",
            "Idle-only bounded windows opened to coalesce prefill arrivals.",
            idle_prefill_formation.clone(),
        );
        registry.register(
            "nml_engine_decode_metadata_rebinds",
            "Stable decode control-slab bindings rebuilt after membership changes.",
            decode_metadata_rebinds.clone(),
        );
        registry.register(
            "nml_engine_active_sequences",
            "Sequences currently owned by the engine scheduler.",
            active.clone(),
        );
        registry.register(
            "nml_engine_queued_requests",
            "Requests waiting for engine execution.",
            queued.clone(),
        );
        registry.register(
            "nml_cache_total_pages",
            "Physical pages in the process-wide target KV arena.",
            cache_total_pages.clone(),
        );
        registry.register(
            "nml_cache_free_pages",
            "Unallocated physical pages in the target KV arena.",
            cache_free_pages.clone(),
        );
        registry.register(
            "nml_cache_reserved_future_pages",
            "Eagerly assigned target-cache pages beyond tentative visibility.",
            cache_reserved_pages.clone(),
        );
        registry.register(
            "nml_engine_decode_batch_rows",
            "Active sequence rows in one submitted decode batch.",
            decode_batch_rows.clone(),
        );
        registry.register(
            "nml_engine_prefill_batch_rows",
            "Active sequence rows in one submitted prefill batch.",
            prefill_batch_rows.clone(),
        );
        registry.register(
            "nml_engine_prefill_batch_tokens",
            "Active prompt tokens in one submitted prefill batch.",
            prefill_batch_tokens.clone(),
        );
        registry.register(
            "nml_engine_decode_batch_seconds",
            "Device submission through compact result download for one decode batch.",
            decode_batch_seconds.clone(),
        );
        registry.register(
            "nml_engine_prefill_batch_seconds",
            "Device submission through compact result download for one prefill batch.",
            prefill_batch_seconds.clone(),
        );
        registry.register(
            "nml_request_ttft_seconds",
            "Admission-to-first-token latency.",
            ttft_seconds.clone(),
        );
        registry.register(
            "nml_request_tpot_seconds",
            "Elapsed time between consecutive raw output tokens.",
            tpot_seconds.clone(),
        );
        registry.register(
            "nml_request_duration_seconds",
            "End-to-end admitted request latency.",
            request_seconds.clone(),
        );
        registry.register(
            "nml_engine_iteration_seconds",
            "One bounded command-drain and inference-step iteration.",
            engine_iteration_seconds.clone(),
        );
        Self {
            registry: Arc::new(Mutex::new(registry)),
            received,
            admitted,
            completed,
            cancelled,
            failed,
            queue_full,
            idle_prefill_formation,
            decode_metadata_rebinds,
            active,
            queued,
            cache_total_pages,
            cache_free_pages,
            cache_reserved_pages,
            decode_batch_rows,
            prefill_batch_rows,
            prefill_batch_tokens,
            decode_batch_seconds,
            prefill_batch_seconds,
            ttft_seconds,
            tpot_seconds,
            request_seconds,
            engine_iteration_seconds,
        }
    }

    pub(crate) fn render(&self) -> Result<String, std::fmt::Error> {
        let mut output = String::new();
        encode(
            &mut output,
            &self
                .registry
                .lock()
                .expect("metrics registry is not poisoned"),
        )?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposition_has_stable_names_and_no_request_labels() {
        let metrics = Metrics::new();
        metrics.received.inc();
        let text = metrics.render().unwrap();
        assert!(text.contains("nml_requests_received_total 1"));
        assert!(text.contains("nml_engine_idle_prefill_formation_total 0"));
        assert!(text.contains("nml_engine_decode_metadata_rebinds_total 0"));
        assert!(text.contains("nml_cache_reserved_future_pages 0"));
        assert!(!text.contains("request_id"));
    }
}
