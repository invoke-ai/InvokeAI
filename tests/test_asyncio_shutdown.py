"""
Tests that verify the fix for the two-Ctrl+C shutdown hang.

Root cause: asyncio.to_thread() (used during generation for SQLite session queue operations)
creates non-daemon threads via the event loop's default ThreadPoolExecutor. When the event
loop is interrupted by KeyboardInterrupt without calling loop.shutdown_default_executor() and
loop.close(), those non-daemon threads remain alive and cause threading._shutdown() to block.

The fix in run_app.py:
1. Cancels all pending asyncio tasks (e.g. socket.io ping tasks) to avoid "Task was destroyed
   but it is pending!" warnings when loop.close() is called.
2. Calls loop.run_until_complete(loop.shutdown_default_executor()) followed by loop.close()
   after ApiDependencies.shutdown(), so all executor threads are cleaned up before the process
   begins its Python-level teardown.
"""

from tests.dangerously_run_function_in_subprocess import dangerously_run_function_in_subprocess


def test_asyncio_to_thread_creates_nondaemon_thread():
    """Confirm that asyncio.to_thread() leaves a non-daemon thread alive after run_until_complete()
    is interrupted - this is the raw symptom that caused the two-Ctrl+C hang."""

    def test_func():
        import asyncio
        import threading

        async def use_thread():
            await asyncio.to_thread(lambda: None)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(use_thread())
        # Deliberately do NOT call shutdown_default_executor() or loop.close()
        non_daemon = [t for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()]
        # There should be at least one non-daemon executor thread still alive
        if not non_daemon:
            raise AssertionError("Expected a non-daemon thread but found none")
        print("ok")

    stdout, _stderr, returncode = dangerously_run_function_in_subprocess(test_func)
    assert returncode == 0, _stderr
    assert stdout.strip() == "ok"


def test_shutdown_default_executor_cleans_up_nondaemon_threads():
    """Verify that calling shutdown_default_executor() + loop.close() eliminates all non-daemon
    threads created by asyncio.to_thread() - this is the fix applied in run_app.py."""

    def test_func():
        import asyncio
        import threading

        async def use_thread():
            await asyncio.to_thread(lambda: None)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(use_thread())

        # Apply the fix
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

        non_daemon = [t for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()]
        if non_daemon:
            raise AssertionError(f"Expected no non-daemon threads but found: {[t.name for t in non_daemon]}")
        print("ok")

    stdout, _stderr, returncode = dangerously_run_function_in_subprocess(test_func)
    assert returncode == 0, _stderr
    assert stdout.strip() == "ok"


def test_shutdown_default_executor_works_after_simulated_keyboard_interrupt():
    """Verify that the fix works even when run_until_complete() was previously interrupted,
    matching the exact flow in run_app.py's except KeyboardInterrupt block."""

    def test_func():
        import asyncio
        import threading

        async def use_thread_then_raise():
            await asyncio.to_thread(lambda: None)
            raise KeyboardInterrupt

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(use_thread_then_raise())
        except KeyboardInterrupt:
            pass

        # At this point a non-daemon thread exists (the bug)
        non_daemon_before = [t for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()]
        if not non_daemon_before:
            raise AssertionError("Expected a non-daemon thread before fix")

        # Apply the fix (what run_app.py now does)
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

        non_daemon_after = [t for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()]
        if non_daemon_after:
            raise AssertionError(f"Non-daemon threads remain after fix: {[t.name for t in non_daemon_after]}")
        print("ok")

    stdout, _stderr, returncode = dangerously_run_function_in_subprocess(test_func)
    assert returncode == 0, _stderr
    assert stdout.strip() == "ok"


def test_cancel_pending_tasks_suppresses_destroyed_task_warnings():
    """Verify that cancelling pending tasks before loop.close() suppresses 'Task was destroyed
    but it is pending!' warnings (e.g. from socket.io ping tasks)."""

    def test_func():
        import asyncio

        async def long_running():
            await asyncio.sleep(1)  # simulates a socket.io ping task

        async def start_background_task():
            asyncio.create_task(long_running())
            await asyncio.to_thread(lambda: None)
            raise KeyboardInterrupt

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(start_background_task())
        except KeyboardInterrupt:
            pass

        # Apply the task-cancellation fix
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()
        print("ok")

    stdout, _stderr, returncode = dangerously_run_function_in_subprocess(test_func)
    assert returncode == 0, _stderr
    assert stdout.strip() == "ok"
    # The "Task was destroyed but it is pending!" message appears on stderr when tasks are NOT
    # cancelled before loop.close(). After the fix it must be absent.
    assert "Task was destroyed but it is pending" not in _stderr
