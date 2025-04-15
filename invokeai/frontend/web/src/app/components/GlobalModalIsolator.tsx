import { GlobalImageHotkeys } from 'app/components/GlobalImageHotkeys';
import ChangeBoardModal from 'features/changeBoardModal/components/ChangeBoardModal';
import { CanvasPasteModal } from 'features/controlLayers/components/CanvasPasteModal';
import {
  NewCanvasSessionDialog,
  NewGallerySessionDialog,
} from 'features/controlLayers/components/NewSessionConfirmationAlertDialog';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import DeleteImageModal from 'features/deleteImageModal/components/DeleteImageModal';
import { FullscreenDropzone } from 'features/dnd/FullscreenDropzone';
import { DynamicPromptsModal } from 'features/dynamicPrompts/components/DynamicPromptsPreviewModal';
import DeleteBoardModal from 'features/gallery/components/Boards/DeleteBoardModal';
import { ImageContextMenu } from 'features/gallery/components/ImageContextMenu/ImageContextMenu';
import { ShareWorkflowModal } from 'features/nodes/components/sidePanel/workflow/WorkflowLibrary/ShareWorkflowModal';
import { WorkflowLibraryModal } from 'features/nodes/components/sidePanel/workflow/WorkflowLibrary/WorkflowLibraryModal';
import { CancelAllExceptCurrentQueueItemConfirmationAlertDialog } from 'features/queue/components/CancelAllExceptCurrentQueueItemConfirmationAlertDialog';
import { ClearQueueConfirmationsAlertDialog } from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { DeleteStylePresetDialog } from 'features/stylePresets/components/DeleteStylePresetDialog';
import { StylePresetModal } from 'features/stylePresets/components/StylePresetForm/StylePresetModal';
import RefreshAfterResetModal from 'features/system/components/SettingsModal/RefreshAfterResetModal';
import { VideosModal } from 'features/system/components/VideosModal/VideosModal';
import { DeleteWorkflowDialog } from 'features/workflowLibrary/components/DeleteLibraryWorkflowConfirmationAlertDialog';
import { LoadWorkflowConfirmationAlertDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { LoadWorkflowFromGraphModal } from 'features/workflowLibrary/components/LoadWorkflowFromGraphModal/LoadWorkflowFromGraphModal';
import { NewWorkflowConfirmationAlertDialog } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import { SaveWorkflowAsDialog } from 'features/workflowLibrary/components/SaveWorkflowAsDialog';
import { memo } from 'react';

/**
 * GlobalModalIsolator is a logical component that isolates global modal components, so that they do not cause needless
 * re-renders of any other components.
 */
export const GlobalModalIsolator = memo(() => {
  return (
    <>
      <DeleteImageModal />
      <ChangeBoardModal />
      <DynamicPromptsModal />
      <StylePresetModal />
      <WorkflowLibraryModal />
      <CancelAllExceptCurrentQueueItemConfirmationAlertDialog />
      <ClearQueueConfirmationsAlertDialog />
      <NewWorkflowConfirmationAlertDialog />
      <LoadWorkflowConfirmationAlertDialog />
      <DeleteStylePresetDialog />
      <DeleteWorkflowDialog />
      <ShareWorkflowModal />
      <RefreshAfterResetModal />
      <DeleteBoardModal />
      <GlobalImageHotkeys />
      <NewGallerySessionDialog />
      <NewCanvasSessionDialog />
      <ImageContextMenu />
      <FullscreenDropzone />
      <VideosModal />
      <SaveWorkflowAsDialog />
      <CanvasManagerProviderGate>
        <CanvasPasteModal />
      </CanvasManagerProviderGate>
      <LoadWorkflowFromGraphModal />
    </>
  );
});
GlobalModalIsolator.displayName = 'GlobalModalIsolator';
