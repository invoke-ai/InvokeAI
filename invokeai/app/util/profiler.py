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
      profiler = Profiler(logger, output_dir)
      # Start a new profile
      profiler.new("my_profile")
      profiler.enable()
      # Do stuff
      profiler.disable()
      profiler.dump()
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

    def __init__(self, logger: Logger, output_dir: Path) -> None:
        self.logger = logger
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler: Optional[cProfile.Profile] = None
        self.profile_id: Optional[str] = None

    def new(self, profile_id: str) -> None:
        self.profiler = cProfile.Profile()
        self.profile_id = profile_id

    def enable(self) -> None:
        """Start profiling."""
        if not self.profiler:
            raise RuntimeError("Profiler not initialized. Call Profiler.new() first.")
        self.profiler.enable()
        self.logger.info(f"Started profiling {self.profile_id}.")

    def disable(self) -> None:
        """Stop profiling."""
        if not self.profiler:
            raise RuntimeError("Profiler not initialized. Call Profiler.new() first.")
        self.profiler.disable()
        self.logger.info(f"Stopped profiling {self.profile_id}.")

    def dump(self) -> None:
        """Dump the profile to disk."""
        if not self.profiler:
            raise RuntimeError("Profiler not initialized. Call Profiler.new() first.")
        self.profiler.dump_stats(self.output_dir / f"{self.profile_id}.prof")
        self.logger.info(f"Dumped profile for {self.profile_id}.")
