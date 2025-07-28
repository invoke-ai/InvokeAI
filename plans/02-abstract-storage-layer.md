# Task: Abstract the Storage Layer

**Status:** Not Started

**Depends On:** 01-decouple-model-backend

---

### 1. Objective

To replace the local file storage with Cloudflare R2 and the local SQLite database with Cloudflare D1.

---

### 2. The "Why"

This is the second major step in transforming InvokeAI into a cloud-native application. By abstracting the storage layer, we remove the dependency on the local filesystem and a local database, making the application stateless and ready for scalable, serverless deployment.

---

### 3. The "How" - Implementation Details

#### **Part 1: Abstracting the File Storage (Cloudflare R2)**

*   **New files to be created:**
    *   `invokeai/app/services/image_files/image_files_r2.py`

*   **Files to be modified:**
    *   `invokeai/app/api/dependencies.py`
    *   `pyproject.toml`

*   **Step-by-step instructions:**

    1.  **Add Cloudflare R2 SDK to `pyproject.toml`:**
        *   Add the `boto3` and `botocore` libraries to the `[project.dependencies]` section. These are the official AWS SDK for Python, which is compatible with Cloudflare R2's S3-compatible API.

    2.  **Create `invokeai/app/services/image_files/image_files_r2.py`:**
        *   Create a new file with a class `R2ImageFileStorage` that implements the `ImageFileStorageBase` interface.
        *   This class will use the `boto3` library to interact with the Cloudflare R2 API.
        *   The constructor will take the R2 bucket name, account ID, access key ID, and access key secret as arguments.
        *   Implement the `get()`, `save()`, and `delete()` methods to perform the corresponding operations on the R2 bucket.
        *   The `get_path()` and `validate_path()` methods will need to be adapted for R2, as they will no longer refer to local file paths. `get_path` could return the R2 object key, and `validate_path` could check for the existence of the object in R2.

    3.  **Modify `invokeai/app/api/dependencies.py`:**
        *   Import the `R2ImageFileStorage` from `invokeai.app.services.image_files.image_files_r2`.
        *   In the `ApiDependencies.initialize()` method, replace the initialization of the `DiskImageFileStorage` with the `R2ImageFileStorage`.
        *   The R2 credentials and bucket name should be read from the application's configuration.

#### **Part 2: Abstracting the Database (Cloudflare D1)**

*   **Files to be modified:**
    *   `invokeai/app/services/shared/sqlite/`
    *   `invokeai/app/api/dependencies.py`

*   **Step-by-step instructions:**

    1.  **Analyze D1 Compatibility:**
        *   Thoroughly review the Cloudflare D1 documentation to understand its SQL dialect and API.
        *   Identify any potential incompatibilities with the existing SQLite-based services.

    2.  **Create D1-Compatible Services:**
        *   Create new versions of the SQLite-based services (e.g., `D1ImageRecordStorage`, `D1SessionQueue`) that are compatible with Cloudflare D1.
        *   This will likely involve replacing the `sqlite3` library with a new library for interacting with the D1 API (e.g., using `requests` or a dedicated D1 library if one exists).
        *   The SQL queries may need to be adjusted to be compatible with the D1 SQL dialect.

    3.  **Modify `invokeai/app/api/dependencies.py`:**
        *   Update the `ApiDependencies.initialize()` method to use the new D1-compatible services.
        *   The D1 credentials and database ID should be read from the application's configuration.

---

### 4. Acceptance Criteria

*   [ ] The `R2ImageFileStorage` is used for all image file operations.
*   [ ] Images are successfully uploaded to and retrieved from Cloudflare R2.
*   [ ] The D1-compatible services are used for all database operations.
*   [ ] The application can start and run without errors using the new storage and database services.
*   [ ] The local `outputs` directory is no longer used for storing images.
*   [ ] The local SQLite database is no longer used.

---

### 5. Notes and Discoveries

*   This section will be filled in as the task is being implemented.

---

### 6. Next Steps

*   [ ] Implement the `R2ImageFileStorage` class.
*   [ ] Implement the D1-compatible database services.
*   [ ] Test the new storage and database services thoroughly.
