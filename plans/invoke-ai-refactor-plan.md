# Plan: Refactor InvokeAI for Cloud-Native Deployment

**Implementation Notes for the AI Assistant:**

This document outlines the plan to refactor the InvokeAI application for a cloud-native, serverless-first architecture. The immediate priorities are:

1.  **Deploy Frontend to Cloudflare:** Deploy the frontend to Cloudflare Pages to make it publicly accessible. See [Task: Deploy Frontend to Cloudflare](plans/00-deploy-frontend-to-cloudflare.md) for details.
2.  **Decouple the Model Backend:** Modify the application to use third-party APIs for image generation instead of local models.
3.  **Abstract the Storage Layer:** Refactor the storage system to use Cloudflare R2 for file storage and Cloudflare D1 for the database.
4.  **Containerize the Backend:** Create a lightweight Docker container for the modified backend, suitable for deployment on Cloudflare's container platform.
</content>

---

This document details the plan to transform the InvokeAI application from a self-hosted, all-in-one solution into a modern, scalable, and cloud-native platform. We will achieve this by decoupling the AI model backend, abstracting the storage layer, and preparing the application for a serverless, container-based deployment.

---

### **1. The Problem: A Monolithic, Self-Hosted Architecture**

The current InvokeAI application is designed to be run on a local machine with a powerful GPU. It's a monolithic application that handles everything from the user interface to the AI model execution. This architecture is not suitable for a cloud-native deployment for several reasons:

*   **High Resource Requirements:** It requires a powerful and expensive server with a GPU to run the AI models.
*   **Not Scalable:** It's difficult to scale the application to handle a large number of users.
*   **Difficult to Maintain:** The monolithic architecture makes it difficult to update and maintain the different parts of the application independently.
*   **Tied to Local Storage:** The application is tightly coupled to the local filesystem for storing images and the database.

---

### **2. The Goal: A Cloud-Native, Serverless-First Architecture**

Our goal is to refactor the application to be:

*   **Lightweight:** The backend should be a lightweight API that doesn't require a GPU.
*   **Scalable:** The application should be able to handle a large number of users without any manual intervention.
*   **Maintainable:** The different parts of the application (frontend, backend, storage) should be independent and easy to update.
*   **Cloud-Native:** The application should be designed to run on modern cloud platforms like Cloudflare.

---

### **3. The Solution: A New, Decoupled Architecture**

To achieve our goals, we will implement a new, decoupled architecture for the InvokeAI application.

#### **3.1. `invokeai/app/api/dependencies.py` - Decoupling the Model Backend**

*   **Objective:** To remove the local AI model loading and execution from the application.
*   **Action:**
    *   Modify the `ApiDependencies.initialize()` method to prevent the `ModelManagerService` from being initialized. We will replace it with a "dummy" service that does nothing.
    *   Remove all dependencies related to local model execution (e.g., `torch`, `diffusers`, `onnx`) from the `pyproject.toml` file.

#### **3.2. `invokeai/app/invocations/` - A New Invocation for Third-Party APIs**

*   **Objective:** To create a new mechanism for generating images using third-party APIs.
*   **Action:**
    *   Create a new invocation (e.g., `ThirdPartyImageGenerationInvocation`) that will be responsible for calling the third-party API.
    *   This invocation will take the user's prompt and other parameters as input.
    *   It will make a request to the third-party API and return the generated image.

#### **3.3. `invokeai/app/services/images/` - Abstracting the File Storage**

*   **Objective:** To replace the local file storage with Cloudflare R2.
*   **Action:**
    *   Create a new `R2ImageFileStorage` class that implements the same interface as the existing `DiskImageFileStorage` class.
    *   This new class will use the Cloudflare R2 API to upload and download images.
    *   Update the `ApiDependencies.initialize()` method to use the new `R2ImageFileStorage` class.

#### **3.4. `invokeai/app/services/shared/sqlite/` - Abstracting the Database**

*   **Objective:** To replace the local SQLite database with Cloudflare D1.
*   **Action:**
    *   Create new versions of the SQLite-based services (e.g., `SqliteImageRecordStorage`, `SqliteSessionQueue`) that are compatible with Cloudflare D1.
    *   This will involve replacing the direct SQLite connections with calls to the Cloudflare D1 API.
    *   Update the `ApiDependencies.initialize()` method to use the new D1-compatible services.

#### **3.5. `Dockerfile` - Creating a Lightweight Container**

*   **Objective:** To create a lightweight Docker container for the modified backend.
*   **Action:**
    *   Modify the existing `Dockerfile` to remove all the dependencies and steps related to local model execution.
    *   The new `Dockerfile` will create a small, efficient container that is perfect for deployment on Cloudflare's container platform.

---

### **4. Benefits of the New Approach**

The new architecture will provide a number of benefits:

*   **Reduced Costs:** We will no longer need to pay for a powerful server with a GPU.
*   **Increased Scalability:** The application will be able to handle a virtually unlimited number of users.
*   **Improved Maintainability:** The decoupled architecture will make it much easier to update and maintain the application.
*   **Future-Proof:** The new architecture is built on modern, serverless technologies that are the future of cloud computing.

---

### **5. Visualizing the New Flow**

Here is a Mermaid diagram that illustrates the new, cloud-native architecture:

```mermaid
graph TD
    subgraph Cloudflare
        A[Frontend on Cloudflare Pages] --> B{Backend in Cloudflare Container};
        B --> C[Third-Party AI API];
        B --> D[Cloudflare R2 for File Storage];
        B --> E[Cloudflare D1 for Database];
    end

    subgraph User
        F[User's Browser] --> A;
    end
