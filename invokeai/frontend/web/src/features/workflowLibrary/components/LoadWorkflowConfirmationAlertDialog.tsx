import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { selectWorkflowIsTouched, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import type { WorkflowV3 } from 'features/nodes/types/workflow';
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

const $workflowToLoad = atom<
  (LoadLibraryWorkflowData & { isOpen: boolean }) | (LoadDirectWorkflowData & { isOpen: boolean }) | null
>(null);
const cleanup = () => $workflowToLoad.set(null);

export const useLoadWorkflow = () => {
  const dispatch = useAppDispatch();
  const workflowLibraryModal = useWorkflowLibraryModal();
  const loadWorkflowFromLibrary = useLoadWorkflowFromLibrary();
  const validatedAndLoadWorkflow = useValidateAndLoadWorkflow();

  const isTouched = useAppSelector(selectWorkflowIsTouched);

  const loadImmediate = useCallback(async () => {
    const data = $workflowToLoad.get();
    if (!data) {
      return;
    }
    if (data.type === 'direct') {
      const validatedWorkflow = await validatedAndLoadWorkflow(data.workflow);
      if (validatedWorkflow) {
        dispatch(workflowModeChanged(data.mode));
      }
    } else {
      await loadWorkflowFromLibrary(data.workflowId, {
        onSuccess: () => {
          dispatch(workflowModeChanged(data.mode));
        },
      });
    }
    cleanup();
    workflowLibraryModal.close();
  }, [dispatch, loadWorkflowFromLibrary, validatedAndLoadWorkflow, workflowLibraryModal]);

  const loadWithDialog = useCallback(
    (data: LoadLibraryWorkflowData | LoadDirectWorkflowData) => {
      if (!isTouched) {
        $workflowToLoad.set({ ...data, isOpen: false });
        loadImmediate();
      } else {
        $workflowToLoad.set({ ...data, isOpen: true });
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
  const workflow = useStore($workflowToLoad);
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
