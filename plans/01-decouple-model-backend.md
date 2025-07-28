# Task: Decouple the Model Backend

**Status:** In Progress

**Depends On:** None

---

### 1. Objective

To remove the local AI model loading and execution from the application and prepare it to use third-party APIs for image generation. This includes ensuring the application can start and run without `torch` or other model-related dependencies.

---

### 2. The "Why"

This is the first and most critical step in transforming InvokeAI into a lightweight, cloud-native application. By removing the dependency on local models, we significantly reduce the resource requirements of the backend, making it possible to deploy it in a scalable, cost-effective containerized environment. Successfully completing this phase is crucial for moving to subsequent refactoring stages.

---

### 3. The "How" - Implementation Details

This task is divided into two phases:

#### **Phase 1: Aggressive Decoupling (The "Rip Out" Phase)**

*   **Objective:** To quickly remove the bulk of the `torch` dependencies to get the application into a runnable state without the model backend.
*   **Files/Directories to be removed:**
    *   `invokeai/backend/model_manager/load/`
    *   `invokeai/backend/model_manager/load/model_loaders/`
    *   `invokeai/backend/model_manager/load/model_cache/`
    *   `invokeai/backend/patches/`
    *   `invokeai/backend/quantization/`
    *   `invokeai/backend/spandrel_image_to_image_model.py`
    *   `invokeai/backend/ip_adapter/resampler.py`
    *   ...and other files identified in "Category 1" of the analysis.
*   **Files to be modified:**
    *   `invokeai/app/services/shared/invocation_context.py`: Remove `torch` import and replace `Tensor` with `Any`.
    *   Other files that might have dangling references to the removed files.

#### **Phase 2: Iterative Refinement**

*   **Objective:** To carefully refactor the remaining parts of the codebase that have `torch` dependencies but are not directly related to model loading.
*   **Files to be modified:**
    *   `invokeai/backend/image_util/**`: Replace `torch`-based image processing with alternatives like `Pillow` or `numpy`.
    *   `invokeai/backend/model_manager/merge.py`: Dummy out the model merging logic.
    *   ...and other files identified in "Category 2" of the analysis.

---

### 4. Acceptance Criteria

*   [ ] The application starts without any `torch`-related `ModuleNotFoundError` errors.
*   [ ] All files and directories in "Category 1" are removed or dummied out.
*   [ ] The `invokeai/app/services/shared/invocation_context.py` file is updated as specified.
*   [ ] The application is significantly lighter and the installation process is faster.

---

### 5. Notes and Discoveries

#### **Message to the Human Project Manager**

The initial analysis underestimated the extent of the `torch` dependencies. A more granular scan of the codebase revealed that the model backend is deeply integrated into many parts of the application.

The new two-phased approach will allow us to make rapid progress by first removing the most problematic components, and then iteratively refining the rest of the codebase. This will give us a runnable application much faster and will make the refactoring process more manageable.

### **Critical Tooling Failures and a Revised Strategy**

#### **Message to the Human Project Manager**

We've hit a significant roadblock in our refactoring efforts due to critical failures in the file editing tools. Specifically, our attempts to modify `invokeai/app/services/shared/invocation_context.py` have been unsuccessful.

1.  **`replace_in_file` Failures:** The `replace_in_file` tool has repeatedly failed to find and replace the `from torch import Tensor` line, despite the line being present in the file. This suggests a potential issue with the tool's matching mechanism or the presence of hidden characters.

2.  **`write_to_file` Corruption:** As a fallback, we attempted to use the `write_to_file` tool. However, this resulted in the corruption of the Python file by appending the `environment_details` to the end of the file, causing a cascade of Pylance errors.

These tooling failures have made it impossible to proceed with the planned file modifications. We need to address this issue before we can continue with the refactoring.

#### **Message to the AI Assistant (Cline)**

Your immediate priority is to find a reliable method for modifying files. The current tools are not working as expected.

**Revised Strategy for File Modifications:**

1.  **Generate Full File Content:** Instead of relying on `replace_in_file`, you will now generate the *entire* content of the file you need to modify in your `thinking` block.
2.  **Use `write_to_file`:** You will then use the `write_to_file` tool to write the complete, corrected content to the file. This will ensure that no extraneous information is appended to the file.
3.  **Verify:** After each file modification, you will read the file back to ensure that the changes were applied correctly and that no new errors were introduced.

This new strategy should allow us to bypass the issues with the file editing tools and get the project back on track.

---

### 6. Next Steps

*   **[UP NEXT]** **Stabilize File Modification Process:** Implement the revised strategy for file modifications to ensure that we can reliably edit files without introducing errors.
*   **[PENDING]** **Execute Phase 1:** Begin the "Aggressive Decoupling" phase.
*   **[PENDING]** **Execute Phase 2:** Begin the "Iterative Refinement" phase.
*   **[PENDING]** **Move to Phase 2 of the master plan:** Once this task is complete, we can move on to the next phase of the refactoring plan: "Abstract the Storage Layer."
