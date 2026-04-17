# Orphaned Models Synchronization Feature

## Overview
This feature adds a UI for synchronizing the models directory by finding and removing orphaned model files. Orphaned models are directories that contain model files but are not referenced in the InvokeAI database.

## Implementation Summary

### Backend (Python)

#### New Service: `OrphanedModelsService`
- Location: `invokeai/app/services/orphaned_models/`
- Implements the core logic from the CLI script
- Methods:
  - `find_orphaned_models()`: Scans the models directory and database to find orphaned models
  - `delete_orphaned_models(paths)`: Safely deletes specified orphaned model directories

#### API Routes
Added to `invokeai/app/api/routers/model_manager.py`:
- `GET /api/v2/models/sync/orphaned`: Returns list of orphaned models with metadata
- `DELETE /api/v2/models/sync/orphaned`: Deletes selected orphaned models

#### Data Models
- `OrphanedModelInfo`: Contains path, absolute_path, files list, and size_bytes
- `DeleteOrphanedModelsRequest`: Contains list of paths to delete
- `DeleteOrphanedModelsResponse`: Contains deleted paths and errors

### Frontend (TypeScript/React)

#### New Components

1. **SyncModelsButton.tsx**
   - Red button styled with `colorScheme="error"` for visual prominence
   - Labeled "Sync Models" 
   - Opens the SyncModelsDialog when clicked
   - Located next to the "+ Add Models" button

2. **SyncModelsDialog.tsx**
   - Modal dialog that displays orphaned models
   - Features:
     - List of orphaned models with checkboxes (default: all checked)
     - "Select All" / "Deselect All" toggle
     - Shows file count and total size for each model
     - "Delete" and "Cancel" buttons
     - Loading spinner while fetching data
     - Error handling with user-friendly messages
   - Automatically shows toast if no orphaned models found
   - Shows success/error toasts after deletion

#### API Integration
- Added `useGetOrphanedModelsQuery` and `useDeleteOrphanedModelsMutation` hooks to `services/api/endpoints/models.ts`
- Integrated with RTK Query for efficient data fetching and caching

#### Translation Strings
Added to `public/locales/en.json`:
- syncModels, noOrphanedModels, orphanedModelsFound
- orphanedModelsDescription, foundOrphanedModels (with pluralization)
- filesCount, deleteSelected, deselectAll
- Success/error messages for deletion operations

## User Experience Flow

1. User clicks the red "Sync Models" button in the Model Manager
2. System queries the backend for orphaned models
3. If no orphaned models:
   - Toast message: "The models directory is synchronized. No orphaned files found."
   - Dialog closes automatically
4. If orphaned models found:
   - Dialog shows list with checkboxes (all selected by default)
   - User can toggle individual models or use "Select All" / "Deselect All"
   - Each model shows:
     - Directory path
     - File count
     - Total size (formatted: B, KB, MB, GB)
5. User clicks "Delete {{count}} selected"
6. System deletes selected models
7. Success/error toasts appear
8. Dialog closes

## Safety Features

1. **Database Backup**: The service creates a backup before any deletion
2. **Selective Deletion**: Users choose which models to delete
3. **Path Validation**: Ensures paths are within the models directory
4. **Error Handling**: Reports which models failed to delete and why
5. **Default Selected**: All models are selected by default for convenience
6. **Confirmation Required**: User must explicitly click Delete

## Technical Details

### Directory-Based Detection
The system treats model paths as directories:
- If database has `model-id/file.safetensors`, the entire `model-id/` directory belongs to that model
- All files and subdirectories within a registered model directory are protected
- Only directories with NO registered models are flagged as orphaned

### Supported File Extensions
- .safetensors
- .ckpt
- .pt
- .pth
- .bin
- .onnx

### Skipped Directories
- .download_cache
- .convert_cache
- \_\_pycache\_\_
- .git

## Testing Recommendations

1. **Test with orphaned models**: 
   - Manually copy a model directory to models folder
   - Verify it appears in the dialog
   - Delete it and verify removal

2. **Test with no orphaned models**:
   - Clean install
   - Verify toast message appears

3. **Test partial selection**:
   - Select only some models
   - Verify only selected ones are deleted

4. **Test error scenarios**:
   - Invalid paths
   - Permission issues
   - Verify error messages are clear

## Files Changed

### Backend
- `invokeai/app/services/orphaned_models/__init__.py` (new)
- `invokeai/app/services/orphaned_models/orphaned_models_service.py` (new)
- `invokeai/app/api/routers/model_manager.py` (modified)

### Frontend
- `invokeai/frontend/web/src/services/api/endpoints/models.ts` (modified)
- `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelManager.tsx` (modified)
- `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelManagerPanel/SyncModelsButton.tsx` (new)
- `invokeai/frontend/web/src/features/modelManagerV2/subpanels/ModelManagerPanel/SyncModelsDialog.tsx` (new)
- `invokeai/frontend/web/public/locales/en.json` (modified)

## Future Enhancements

Potential improvements for future versions:
1. Show preview of what will be deleted before deletion
2. Add option to move orphaned models to archive instead of deleting
3. Show disk space that will be freed
4. Add filter/search in orphaned models list
5. Support for undo operation
6. Scheduled automatic cleanup
