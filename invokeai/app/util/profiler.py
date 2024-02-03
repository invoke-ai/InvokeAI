import cProfile
from logging import Logger
from pathlib import Path
from typing import Optional


class Profiler:
    """
    Simple wrapper around cProfile.

    Usage
    ```
      # Create a profiler
      profiler = Profiler(logger, output_dir, "sql_query_perf")
      # Start a new profile
      profiler.start("my_profile")
      # Do stuff
      profiler.stop()
    ```

    Visualize a profile as a flamegraph with [snakeviz](https://jiffyclub.github.io/snakeviz/)
    ```sh
      snakeviz my_profile.prof
    ```

    Visualize a profile as directed graph with [graphviz](https://graphviz.org/download/) & [gprof2dot](https://github.com/jrfonseca/gprof2dot)
    ```sh
      gprof2dot -f pstats my_profile.prof | dot -Tpng -o my_profile.png
      # SVG or PDF may be nicer - you can search for function names
      gprof2dot -f pstats my_profile.prof | dot -Tsvg -o my_profile.svg
      gprof2dot -f pstats my_profile.prof | dot -Tpdf -o my_profile.pdf
    ```
    """

    def __init__(self, logger: Logger, output_dir: Path, prefix: Optional[str] = None) -> None:
        self._logger = logger.getChild(f"profiler.{prefix}" if prefix else "profiler")
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._profiler: Optional[cProfile.Profile] = None
        self._prefix = prefix

        self.profile_id: Optional[str] = None

    def start(self, profile_id: str) -> None:
        if self._profiler:
            self.stop()

        self.profile_id = profile_id

        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._logger.info(f"Started profiling {self.profile_id}.")

    def stop(self) -> Path:
        if not self._profiler:
            raise RuntimeError("Profiler not initialized. Call start() first.")
        self._profiler.disable()

        filename = f"{self._prefix}_{self.profile_id}.prof" if self._prefix else f"{self.profile_id}.prof"
        path = Path(self._output_dir, filename)

        self._profiler.dump_stats(path)
        self._logger.info(f"Stopped profiling, profile dumped to {path}.")
        self._profiler = None
        self.profile_id = None

        return path
