import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { selectWorkflowIsTouched } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { useLoadWorkflowFromImage } from 'features/workflowLibrary/hooks/useLoadWorkflowFromImage';
import { useLoadWorkflowFromLibrary } from 'features/workflowLibrary/hooks/useLoadWorkflowFromLibrary';
import { useLoadWorkflowFromObject } from 'features/workflowLibrary/hooks/useLoadWorkflowFromObject';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Callbacks = {
  onSuccess?: (workflow: WorkflowV3) => void;
  onError?: () => void;
  onCompleted?: () => void;
};

type LoadLibraryWorkflowData = Callbacks & {
  type: 'library';
  data: string;
};

type LoadWorkflowFromObjectData = Callbacks & {
  type: 'object';
  data: unknown;
};

type LoadWorkflowFromFileData = Callbacks & {
  type: 'file';
  data: File;
};

type LoadWorkflowFromImageData = Callbacks & {
  type: 'image';
  data: string;
};

type DialogStateExtra = {
  isOpen: boolean;
};

const $dialogState = atom<
  | (LoadLibraryWorkflowData & DialogStateExtra)
  | (LoadWorkflowFromObjectData & DialogStateExtra)
  | (LoadWorkflowFromFileData & DialogStateExtra)
  | (LoadWorkflowFromImageData & DialogStateExtra)
  | null
>(null);
const cleanup = () => $dialogState.set(null);

const useLoadImmediate = () => {
  const workflowLibraryModal = useWorkflowLibraryModal();
  const loadWorkflowFromLibrary = useLoadWorkflowFromLibrary();
  const loadWorkflowFromFile = useLoadWorkflowFromFile();
  const loadWorkflowFromImage = useLoadWorkflowFromImage();
  const loadWorkflowFromObject = useLoadWorkflowFromObject();

  const loadImmediate = useCallback(async () => {
    const dialogState = $dialogState.get();
    if (!dialogState) {
      return;
    }
    const { type, data, onSuccess, onError, onCompleted } = dialogState;
    const options = {
      onSuccess,
      onError,
      onCompleted,
    };
    if (type === 'object') {
      await loadWorkflowFromObject(data, options);
    } else if (type === 'file') {
      await loadWorkflowFromFile(data, options);
    } else if (type === 'library') {
      await loadWorkflowFromLibrary(data, options);
    } else if (type === 'image') {
      await loadWorkflowFromImage(data, options);
    }
    cleanup();
    workflowLibraryModal.close();
  }, [
    loadWorkflowFromFile,
    loadWorkflowFromImage,
    loadWorkflowFromLibrary,
    loadWorkflowFromObject,
    workflowLibraryModal,
  ]);

  return loadImmediate;
};

/**
 * Handles loading workflows from various sources. If there are unsaved changes, the user will be prompted to confirm
 * before loading the workflow.
 */
export const useLoadWorkflowWithDialog = () => {
  const isTouched = useAppSelector(selectWorkflowIsTouched);
  const loadImmediate = useLoadImmediate();

  const loadWorkflowWithDialog = useCallback(
    /**
     * Loads a workflow from various sources. If there are unsaved changes, the user will be prompted to confirm before
     * loading the workflow. The workflow will be loaded immediately if there are no unsaved changes. On success, error
     * or completion, the corresponding callback will be called.
     *
     * @param data - The data to load the workflow from.
     * @param data.type - The type of data to load the workflow from.
     * @param data.data - The data to load the workflow from. The type of this data depends on the `type` field.
     * @param data.onSuccess - A callback to call when the workflow is successfully loaded.
     * @param data.onError - A callback to call when an error occurs while loading the workflow.
     * @param data.onCompleted - A callback to call when the loading process is completed (both success and error).
     */
    (
      data: LoadLibraryWorkflowData | LoadWorkflowFromObjectData | LoadWorkflowFromFileData | LoadWorkflowFromImageData
    ) => {
      if (!isTouched) {
        $dialogState.set({ ...data, isOpen: false });
        loadImmediate();
      } else {
        $dialogState.set({ ...data, isOpen: true });
      }
    },
    [loadImmediate, isTouched]
  );

  return loadWorkflowWithDialog;
};

export const LoadWorkflowConfirmationAlertDialog = memo(() => {
  useAssertSingleton('LoadWorkflowConfirmationAlertDialog');
  const { t } = useTranslation();
  const workflow = useStore($dialogState);
  const loadImmediate = useLoadImmediate();

  return (
    <ConfirmationAlertDialog
      isOpen={!!workflow?.isOpen}
      onClose={cleanup}
      title={t('nodes.loadWorkflow')}
      acceptCallback={loadImmediate}
      useInert={false}
      acceptButtonText={t('common.load')}
    >
      <Flex flexDir="column" gap={2}>
        <Text>{t('nodes.loadWorkflowDesc')}</Text>
        <Text variant="subtext">{t('nodes.loadWorkflowDesc2')}</Text>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

LoadWorkflowConfirmationAlertDialog.displayName = 'LoadWorkflowConfirmationAlertDialog';
