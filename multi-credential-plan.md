# Plan: Graceful Rate-Limit Handling with Credential Cycling

**Implementation Notes for the AI Assistant:**

This document has been updated to address a flaw in the initial implementation. The immediate priority is to fix the authentication retry loop. Please implement the following:

1.  **Update `performRequest` in `src/gemini-client.ts`:** Modify the handling of 401/403 errors to include a retry limit and exponential backoff, as detailed in the updated "Automatic Credential Refresh" section (3.6).
2.  **Verify the new logic:** Ensure that credentials that repeatedly fail to refresh are correctly marked as rate-limited to prevent them from being used again.
---

This document outlines the plan to refactor the application to intelligently handle API rate limits and authentication failures, ensuring a seamless experience for the end user. We will achieve this by creating a robust retry mechanism that automatically cycles through available credentials when a rate limit or authentication error is encountered.

---

This document outlines the plan to refactor the application to intelligently handle API rate limits, ensuring a seamless experience for the end user. We will achieve this by creating a robust retry mechanism that automatically cycles through available credentials when a rate limit is encountered.

---

### **1. The Problem: Unhandled 429 Errors**

The application currently streams 429 rate limit errors directly to the end user, which is a jarring and unprofessional experience. As seen in the `worker-logs.md`, when a credential hits its API quota, the application fails to handle the error gracefully. This not only disrupts the user's workflow but also fails to leverage the multi-credential architecture that was put in place to prevent this exact scenario.

---

### **2. Current Flawed Implementation**

While the `AuthManager` in `src/auth.ts` has a basic catch for 429 errors, it's not sufficient. The core issue is that the error is not being propagated in a way that allows the `GeminiApiClient` to react. The `callEndpoint` method in `AuthManager` is too generic, and the responsibility for handling API-specific errors should lie with the `GeminiApiClient`.

---

### **3. The Solution: A Robust New Architecture**

To address this, we will implement a more robust and resilient architecture for handling API requests and rate limits.

#### **3.1. `src/types.ts` - Refining the Data Structure**

-   **Objective:** To create a more explicit and reliable way to track the rate-limit status of each credential.
-   **Action:**
    -   In the `CredentialStatus` interface, we will modify the `rateLimit` property to be more structured. It will now include an `isRateLimited` boolean flag and an `expiresAt` timestamp.

#### **3.2. `src/credential-manager.ts` - Smarter Credential Management**

-   **Objective:** To give the `GeminiApiClient` a clear and efficient way to get a list of usable credentials.
-   **Action:**
    -   We will introduce a new public method, `getAvailableCredentials(model: string)`, which will return an array of all credentials that are not currently rate-limited for the specified model.
    -   We will refactor the `markCredentialRateLimited` method to be more precise. It will now set the `isRateLimited` flag to `true` and use the `expiresAt` timestamp to determine when the credential should be used again.

#### **3.3. `src/auth.ts` - Streamlining the `AuthManager`**

-   **Objective:** To centralize all API call logic within the `GeminiApiClient` for better separation of concerns.
-   **Action:**
    -   The `callEndpoint` method will be completely removed from the `AuthManager`. Its responsibilities will be absorbed by the `GeminiApiClient`.

#### **3.4. `src/gemini-client.ts` - The Core of the Solution**

-   **Objective:** To implement a robust, self-healing mechanism for handling API requests and rate limits.
-   **Action:**
    -   The `performRequest` method will be updated to include intelligent error handling.
    -   The `streamContent` and `getCompletion` methods will use this new `performRequest` method.

#### **3.5. Intelligent Error Handling**

-   **Objective:** To create a sophisticated error-handling mechanism that can differentiate between different types of errors and take the appropriate action for each.
-   **Action:**
    -   **Rate-Limit Errors (429):** When a 429 error is received, the credential will be marked as rate-limited, and the system will automatically try the next available credential.
    -   **Server Errors (5xx):** For temporary server errors, the system will retry the request with the same credential up to three times with an exponential backoff. If the error persists, the credential will be marked as rate-limited, and the system will move to the next credential.
    -   **Client Errors (4xx):** For client-side errors (e.g., a bad request), the system will fail immediately and report the error, as retrying would be futile.

#### **3.6. Automatic Credential Refresh**

-   **Objective:** To create a self-healing system that can automatically refresh expired credentials without disrupting the user's workflow.
-   **Action:**
    -   A new public method, `refreshCredential(credentialId: string, refreshToken: string)`, will be created in the `AuthManager`.
    -   The `performRequest` method in the `GeminiApiClient` will be updated to catch `401 Unauthorized` and `403 Forbidden` errors.
    -   When one of these errors is caught, the `performRequest` method will:
        -   Attempt to refresh the token using the `refreshCredential` method up to three times with an exponential backoff.
        -   If the refresh is successful, it will retry the original request with the new token.
        -   If the refresh fails after three attempts, the credential will be marked as rate-limited for one hour, and the system will move on to the next available credential.

---

### **4. Benefits of the New Approach**

The new architecture will provide a seamless and professional user experience by eliminating jarring errors. The new implementation will be more resilient and easier to maintain, as the error handling and credential management logic will be centralized and more robust.

---

### **5. Visualizing the New Flow**

Here is a Mermaid diagram that illustrates the new, more robust flow of control, including the new server error retry logic and the updated credential refresh mechanism:

```mermaid
sequenceDiagram
    participant User
    participant RouteHandler
    participant GeminiApiClient
    participant CredentialManager
    participant AuthManager
    participant GoogleAPI

    User->>+RouteHandler: /v1/chat/completions
    RouteHandler->>+GeminiApiClient: streamContent()
    GeminiApiClient->>+CredentialManager: getAvailableCredentials()
    CredentialManager-->>-GeminiApiClient: [Credential1, Credential2]
    
    GeminiApiClient->>+GoogleAPI: API Request with Credential1
    GoogleAPI-->>-GeminiApiClient: 401 Unauthorized

    loop 3 Times with Exponential Backoff
        GeminiApiClient->>+AuthManager: refreshCredential(Credential1)
        AuthManager-->>-GeminiApiClient: Refresh Fails
    end

    GeminiApiClient->>+CredentialManager: markCredentialRateLimited(Credential1)
    GeminiApiClient->>+GoogleAPI: API Request with Credential2
    GoogleAPI-->>-GeminiApiClient: Success (200 OK)
    GeminiApiClient-->>-RouteHandler: Stream Response
    RouteHandler-->>-User: Stream Response
