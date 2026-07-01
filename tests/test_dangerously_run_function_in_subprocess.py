from tests.dangerously_run_function_in_subprocess import dangerously_run_function_in_subprocess


def test_simple_function():
    def test_func():
        print("Hello, Test!")

    stdout, stderr, returncode = dangerously_run_function_in_subprocess(test_func)

    assert returncode == 0
    assert stdout.strip() == "Hello, Test!"
    assert stderr == ""


def test_function_with_error():
    def test_func():
        raise ValueError("This is an error")

    _stdout, stderr, returncode = dangerously_run_function_in_subprocess(test_func)

    assert returncode != 0  # Should fail
    assert "ValueError: This is an error" in stderr


def test_function_with_imports():
    def test_func():
        import math

        print(math.sqrt(4))

    stdout, stderr, returncode = dangerously_run_function_in_subprocess(test_func)

    assert returncode == 0
    assert stdout.strip() == "2.0"
    assert stderr == ""


def test_function_with_sys_exit():
    def test_func():
        import sys

        sys.exit(42)

    _stdout, _stderr, returncode = dangerously_run_function_in_subprocess(test_func)

    assert returncode == 42  # Should return the custom exit code


def test_function_with_closure():
    foo = "bar"

    def test_func():
        print(foo)

    _stdout, _stderr, returncode = dangerously_run_function_in_subprocess(test_func)

    assert returncode == 1  # Should fail because of closure
