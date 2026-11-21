from scripts.benchmark_spark_model_cache_copy import _get_current_record


def test_reacquires_evicted_record_before_switch() -> None:
    class Cache:
        def __init__(self) -> None:
            self.records: dict[str, str] = {}
            self.get_keys: list[str] = []

        def get(self, key: str) -> str:
            self.get_keys.append(key)
            try:
                return self.records[key]
            except KeyError as exc:
                raise IndexError(key) from exc

        def put(self, key: str, value: str) -> None:
            self.records.clear()
            self.records[key] = value

    cache = Cache()
    created: list[str] = []

    def create(name: str) -> str:
        created.append(name)
        return f"record-{name}-{created.count(name)}"

    assert _get_current_record(cache, "a", lambda: create("a")) == "record-a-1"
    assert _get_current_record(cache, "b", lambda: create("b")) == "record-b-1"
    assert _get_current_record(cache, "a", lambda: create("a")) == "record-a-2"
    assert created == ["a", "b", "a"]
    assert cache.get_keys == ["a", "a", "b", "b", "a", "a"]
