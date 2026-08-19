from policy.inference_timing import timed_inference


class DeferredValue:
    def __init__(self):
        self.was_blocked = False

    def block_until_ready(self):
        self.was_blocked = True


def test_timed_inference_blocks_nested_deferred_values():
    deferred = DeferredValue()

    value, elapsed = timed_inference(lambda: {"actions": [deferred]})

    assert value == {"actions": [deferred]}
    assert deferred.was_blocked
    assert elapsed >= 0
