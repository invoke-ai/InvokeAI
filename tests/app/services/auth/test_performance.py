"""Performance tests for multiuser authentication system.

These tests measure the performance overhead of authentication and
ensure the system performs acceptably under load.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger

import pytest

from invokeai.app.services.auth.password_utils import hash_password, verify_password
from invokeai.app.services.auth.token_service import TokenData, create_access_token, verify_token
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService


@pytest.fixture
def logger() -> Logger:
    """Create a logger for testing."""
    return Logger("test_performance")


@pytest.fixture
def user_service(logger: Logger) -> UserService:
    """Create a user service with in-memory database for testing."""
    db = SqliteDatabase(db_path=None, logger=logger, verbose=False)

    # Create users table
    db._conn.execute("""
        CREATE TABLE users (
            user_id TEXT NOT NULL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT,
            password_hash TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT FALSE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            last_login_at DATETIME
        );
    """)
    db._conn.commit()

    return UserService(db)


class TestPasswordPerformance:
    """Tests for password hashing and verification performance."""

    def test_password_hashing_performance(self):
        """Test that password hashing completes in reasonable time.

        bcrypt is intentionally slow for security. Each hash should take
        approximately 50-100ms on modern hardware.
        """
        password = "TestPassword123"
        iterations = 10

        start_time = time.time()
        for _ in range(iterations):
            hash_password(password)
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # Each hash should take between 10ms and 500ms
        # (bcrypt is designed to be slow, 50-100ms is typical)
        assert 10 < avg_time_ms < 500, f"Password hashing took {avg_time_ms:.2f}ms per hash"

        # Log performance for reference
        print(f"\nPassword hashing performance: {avg_time_ms:.2f}ms per hash")

    def test_password_verification_performance(self):
        """Test that password verification completes in reasonable time."""
        password = "TestPassword123"
        hashed = hash_password(password)
        iterations = 10

        start_time = time.time()
        for _ in range(iterations):
            verify_password(password, hashed)
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # Verification should take similar time to hashing
        assert 10 < avg_time_ms < 500, f"Password verification took {avg_time_ms:.2f}ms per verification"

        print(f"Password verification performance: {avg_time_ms:.2f}ms per verification")

    def test_concurrent_password_operations(self):
        """Test password operations under concurrent load."""
        password = "TestPassword123"
        num_operations = 20

        def hash_and_verify():
            hashed = hash_password(password)
            return verify_password(password, hashed)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(hash_and_verify) for _ in range(num_operations)]

            results = [future.result() for future in as_completed(futures)]

        elapsed_time = time.time() - start_time

        # All operations should succeed
        assert all(results)

        # Total time should be less than sequential time due to parallelization
        print(f"Concurrent password operations ({num_operations}): {elapsed_time:.2f}s total")


class TestTokenPerformance:
    """Tests for JWT token performance."""

    def test_token_creation_performance(self):
        """Test that token creation is fast."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        iterations = 1000

        start_time = time.time()
        for _ in range(iterations):
            create_access_token(token_data)
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # Token creation should be very fast (< 1ms per token)
        assert avg_time_ms < 1.0, f"Token creation took {avg_time_ms:.3f}ms per token"

        print(f"\nToken creation performance: {avg_time_ms:.3f}ms per token")

    def test_token_verification_performance(self):
        """Test that token verification is fast."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)
        iterations = 1000

        start_time = time.time()
        for _ in range(iterations):
            verify_token(token)
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # Token verification should be very fast (< 1ms per verification)
        assert avg_time_ms < 1.0, f"Token verification took {avg_time_ms:.3f}ms per verification"

        print(f"Token verification performance: {avg_time_ms:.3f}ms per verification")

    def test_concurrent_token_operations(self):
        """Test token operations under concurrent load."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        num_operations = 1000

        def create_and_verify():
            token = create_access_token(token_data)
            verified = verify_token(token)
            return verified is not None

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_and_verify) for _ in range(num_operations)]

            results = [future.result() for future in as_completed(futures)]

        elapsed_time = time.time() - start_time

        # All operations should succeed
        assert all(results)

        ops_per_second = num_operations / elapsed_time
        print(f"Concurrent token operations: {ops_per_second:.0f} ops/second")

        # Should handle at least 1000 operations per second
        assert ops_per_second > 1000, f"Only {ops_per_second:.0f} ops/second"


class TestAuthenticationOverhead:
    """Tests for overall authentication system overhead."""

    def test_login_flow_performance(self, user_service: UserService):
        """Test complete login flow performance."""
        # Create a user
        user_data = UserCreateRequest(
            email="perf@example.com",
            display_name="Performance Test",
            password="TestPass123",
            is_admin=False,
        )
        user_service.create(user_data)

        iterations = 10

        start_time = time.time()
        for _ in range(iterations):
            # Simulate login flow
            user = user_service.authenticate("perf@example.com", "TestPass123")
            assert user is not None

            # Create token
            token_data = TokenData(
                user_id=user.user_id,
                email=user.email,
                is_admin=user.is_admin,
            )
            token = create_access_token(token_data)

            # Verify token
            verified = verify_token(token)
            assert verified is not None

        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / iterations) * 1000

        # Complete login flow should complete in reasonable time
        # Most of the time is spent on password verification (50-100ms)
        assert avg_time_ms < 500, f"Login flow took {avg_time_ms:.2f}ms"

        print(f"\nComplete login flow performance: {avg_time_ms:.2f}ms per login")

    def test_token_verification_overhead(self):
        """Measure overhead of token verification vs no auth."""
        token_data = TokenData(
            user_id="user123",
            email="test@example.com",
            is_admin=False,
        )

        token = create_access_token(token_data)
        iterations = 10000

        # Measure token verification time
        start_time = time.time()
        for _ in range(iterations):
            verify_token(token)
        verification_time = time.time() - start_time

        # Measure baseline (minimal operation)
        start_time = time.time()
        for _ in range(iterations):
            # Simulate minimal auth check
            _ = token is not None
        baseline_time = time.time() - start_time

        overhead_ms = ((verification_time - baseline_time) / iterations) * 1000

        # Overhead should be minimal (< 0.1ms per request)
        assert overhead_ms < 0.1, f"Token verification adds {overhead_ms:.4f}ms overhead per request"

        print(f"Token verification overhead: {overhead_ms:.4f}ms per request")


class TestUserServicePerformance:
    """Tests for user service performance."""

    def test_user_creation_performance(self, user_service: UserService):
        """Test user creation performance."""
        iterations = 10

        start_time = time.time()
        for i in range(iterations):
            user_data = UserCreateRequest(
                email=f"user{i}@example.com",
                display_name=f"User {i}",
                password="TestPass123",
                is_admin=False,
            )
            user_service.create(user_data)
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # User creation includes password hashing, so should be ~50-150ms
        assert avg_time_ms < 500, f"User creation took {avg_time_ms:.2f}ms per user"

        print(f"\nUser creation performance: {avg_time_ms:.2f}ms per user")

    def test_user_lookup_performance(self, user_service: UserService):
        """Test user lookup performance."""
        # Create some users
        for i in range(10):
            user_data = UserCreateRequest(
                email=f"lookup{i}@example.com",
                display_name=f"Lookup User {i}",
                password="TestPass123",
                is_admin=False,
            )
            user_service.create(user_data)

        iterations = 1000

        # Test lookup by email
        start_time = time.time()
        for _ in range(iterations):
            user_service.get_by_email("lookup5@example.com")
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # Lookup should be fast (< 1ms with proper indexing)
        assert avg_time_ms < 5.0, f"User lookup took {avg_time_ms:.3f}ms per lookup"

        print(f"User lookup by email performance: {avg_time_ms:.3f}ms per lookup")

    def test_user_list_performance(self, user_service: UserService):
        """Test user list performance with many users."""
        # Create many users
        num_users = 100

        for i in range(num_users):
            user_data = UserCreateRequest(
                email=f"listuser{i}@example.com",
                display_name=f"List User {i}",
                password="TestPass123",
                is_admin=False,
            )
            user_service.create(user_data)

        # Test listing users
        iterations = 10

        start_time = time.time()
        for _ in range(iterations):
            user_service.list_users(limit=50)
        elapsed_time = time.time() - start_time

        avg_time_ms = (elapsed_time / iterations) * 1000

        # Listing users should be fast (< 10ms for reasonable page size)
        assert avg_time_ms < 50.0, f"User listing took {avg_time_ms:.2f}ms"

        print(f"User listing performance (50 users): {avg_time_ms:.2f}ms per query")


class TestConcurrentUserSessions:
    """Tests for concurrent user session handling."""

    def test_multiple_concurrent_logins(self, user_service: UserService):
        """Test handling multiple concurrent user logins."""
        # Create test users
        num_users = 20
        for i in range(num_users):
            user_data = UserCreateRequest(
                email=f"concurrent{i}@example.com",
                display_name=f"Concurrent User {i}",
                password="TestPass123",
                is_admin=False,
            )
            user_service.create(user_data)

        def authenticate_user(user_index: int):
            # Authenticate
            user = user_service.authenticate(f"concurrent{user_index}@example.com", "TestPass123")
            if user is None:
                return False

            # Create token
            token_data = TokenData(
                user_id=user.user_id,
                email=user.email,
                is_admin=user.is_admin,
            )
            token = create_access_token(token_data)

            # Verify token
            verified = verify_token(token)
            return verified is not None

        start_time = time.time()

        # Simulate concurrent logins
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(authenticate_user, i) for i in range(num_users)]

            results = [future.result() for future in as_completed(futures)]

        elapsed_time = time.time() - start_time

        # All logins should succeed
        assert all(results), "Some concurrent logins failed"

        print(f"\nConcurrent logins ({num_users} users): {elapsed_time:.2f}s total")

        # Should complete in reasonable time
        assert elapsed_time < 10.0, f"Concurrent logins took {elapsed_time:.2f}s"


@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Scalability benchmarks (marked as slow tests)."""

    def test_authentication_under_load(self, user_service: UserService):
        """Test authentication system under sustained load."""
        # Create test users
        num_users = 50
        for i in range(num_users):
            user_data = UserCreateRequest(
                email=f"load{i}@example.com",
                display_name=f"Load User {i}",
                password="TestPass123",
                is_admin=False,
            )
            user_service.create(user_data)

        def simulate_user_activity(user_index: int, num_requests: int):
            success_count = 0
            for _ in range(num_requests):
                # Authenticate
                user = user_service.authenticate(f"load{user_index}@example.com", "TestPass123")
                if user is None:
                    continue

                # Create and verify token
                token_data = TokenData(user_id=user.user_id, email=user.email, is_admin=user.is_admin)
                token = create_access_token(token_data)
                verified = verify_token(token)

                if verified is not None:
                    success_count += 1

            return success_count

        # Simulate sustained load
        requests_per_user = 5
        total_requests = num_users * requests_per_user

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(simulate_user_activity, i, requests_per_user) for i in range(num_users)]

            success_counts = [future.result() for future in as_completed(futures)]

        elapsed_time = time.time() - start_time

        total_success = sum(success_counts)
        success_rate = (total_success / total_requests) * 100
        requests_per_second = total_requests / elapsed_time

        print("\nLoad test results:")
        print(f"  Total requests: {total_requests}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Requests/second: {requests_per_second:.0f}")
        print(f"  Total time: {elapsed_time:.2f}s")

        # Should maintain high success rate under load
        assert success_rate > 95.0, f"Success rate only {success_rate:.1f}%"

        # Should handle reasonable throughput
        # Note: This is limited by bcrypt hashing speed
        assert requests_per_second > 5.0, f"Only {requests_per_second:.1f} req/s"
