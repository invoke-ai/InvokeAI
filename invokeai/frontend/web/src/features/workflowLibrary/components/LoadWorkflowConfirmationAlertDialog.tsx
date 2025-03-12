import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { selectWorkflowIsTouched, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { useLoadWorkflowFromLibrary } from 'features/workflowLibrary/hooks/useLoadWorkflowFromLibrary';
import { useValidateAndLoadWorkflow } from 'features/workflowLibrary/hooks/useValidateAndLoadWorkflow';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type LoadLibraryWorkflowData = {
  type: 'library';
  workflowId: string;
  mode: 'view' | 'edit';
};

type LoadDirectWorkflowData = {
  type: 'direct';
  workflow: WorkflowV3;
  mode: 'view' | 'edit';
};

type LoadFileWorkflowData = {
  type: 'file';
  file: File;
  mode: 'view' | 'edit';
};

const $dialogState = atom<
  | (LoadLibraryWorkflowData & { isOpen: boolean })
  | (LoadDirectWorkflowData & { isOpen: boolean })
  | (LoadFileWorkflowData & { isOpen: boolean })
  | null
>(null);
const cleanup = () => $dialogState.set(null);

export const useLoadWorkflow = () => {
  const dispatch = useAppDispatch();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const loadWorkflowFromLibrary = useLoadWorkflowFromLibrary();
  const loadWorkflowFromFile = useLoadWorkflowFromFile();
  const validatedAndLoadWorkflow = useValidateAndLoadWorkflow();

  const isTouched = useAppSelector(selectWorkflowIsTouched);

  const loadImmediate = useCallback(async () => {
    const data = $dialogState.get();
    if (!data) {
      return;
    }
    if (data.type === 'direct') {
      const validatedWorkflow = await validatedAndLoadWorkflow(data.workflow);
      if (validatedWorkflow) {
        dispatch(workflowModeChanged(data.mode));
      }
    } else if (data.type === 'file') {
      await loadWorkflowFromFile(data.file, {
        onSuccess: () => {
          dispatch(workflowModeChanged(data.mode));
        },
      });
    } else {
      await loadWorkflowFromLibrary(data.workflowId, {
        onSuccess: () => {
          dispatch(workflowModeChanged(data.mode));
        },
      });
    }
    cleanup();
    workflowLibraryModal.close();
  }, [dispatch, loadWorkflowFromFile, loadWorkflowFromLibrary, validatedAndLoadWorkflow, workflowLibraryModal]);

  const loadWithDialog = useCallback(
    (data: LoadLibraryWorkflowData | LoadDirectWorkflowData | LoadFileWorkflowData) => {
      if (!isTouched) {
        $dialogState.set({ ...data, isOpen: false });
        loadImmediate();
      } else {
        $dialogState.set({ ...data, isOpen: true });
      }
    },
    [loadImmediate, isTouched]
  );

  return {
    loadImmediate,
    loadWithDialog,
  } as const;
};

export const LoadWorkflowConfirmationAlertDialog = memo(() => {
  useAssertSingleton('LoadWorkflowConfirmationAlertDialog');
  const { t } = useTranslation();
  const workflow = useStore($dialogState);
  const loadWorkflow = useLoadWorkflow();

  return (
    <ConfirmationAlertDialog
      isOpen={!!workflow?.isOpen}
      onClose={cleanup}
      title={t('nodes.loadWorkflow')}
      acceptCallback={loadWorkflow.loadImmediate}
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
