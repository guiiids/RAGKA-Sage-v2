import threading
import logging
from collections import defaultdict

# Thread-safe storage for metrics
_lock = threading.Lock()
_latency_data = defaultdict(list)
_token_data = defaultdict(lambda: {'prompt': [], 'completion': []})

# Configure metrics logger to write to metrics.log
_logger = logging.getLogger('metrics')
if not _logger.handlers:
    handler = logging.FileHandler('metrics.log')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


def record_latency(endpoint: str, ms: float) -> None:
    """Record latency for a given endpoint."""
    with _lock:
        _latency_data[endpoint].append(ms)
    _logger.info(f"latency endpoint={endpoint} ms={ms}")


def record_tokens(model: str, prompt: int, completion: int) -> None:
    """Record prompt and completion tokens for a model."""
    with _lock:
        data = _token_data[model]
        data['prompt'].append(prompt)
        data['completion'].append(completion)
    _logger.info(
        f"tokens model={model} prompt={prompt} completion={completion}"
    )


def report_metrics() -> dict:
    """Return average latency and token usage metrics."""
    with _lock:
        latency_report = {
            ep: (sum(vals) / len(vals) if vals else 0.0)
            for ep, vals in _latency_data.items()
        }
        token_report = {}
        for model, vals in _token_data.items():
            prompt_vals = vals['prompt']
            completion_vals = vals['completion']
            token_report[model] = {
                'avg_prompt': sum(prompt_vals) / len(prompt_vals) if prompt_vals else 0.0,
                'avg_completion': sum(completion_vals) / len(completion_vals) if completion_vals else 0.0,
            }
    return {'latency': latency_report, 'tokens': token_report}

