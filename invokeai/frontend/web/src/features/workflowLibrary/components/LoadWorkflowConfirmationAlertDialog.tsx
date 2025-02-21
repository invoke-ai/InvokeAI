import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowIsTouched, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const $workflowToLoad = atom<{ workflowId: string; mode: 'view' | 'edit'; isOpen: boolean } | null>(null);
const cleanup = () => $workflowToLoad.set(null);

export const useLoadWorkflow = () => {
  const dispatch = useAppDispatch();
  const workflowListMenu = useWorkflowListMenu();
  const { getAndLoadWorkflow } = useGetAndLoadLibraryWorkflow();

  const isTouched = useAppSelector(selectWorkflowIsTouched);

  const loadImmediate = useCallback(async () => {
    const workflow = $workflowToLoad.get();
    if (!workflow) {
      return;
    }
    const { workflowId, mode } = workflow;
    await getAndLoadWorkflow(workflowId);
    dispatch(workflowModeChanged(mode));
    cleanup();
    workflowListMenu.close();
  }, [dispatch, getAndLoadWorkflow, workflowListMenu]);

  const loadWithDialog = useCallback(
    (workflowId: string, mode: 'view' | 'edit') => {
      if (!isTouched) {
        $workflowToLoad.set({
          workflowId,
          mode,
          isOpen: false,
        });
        loadImmediate();
      } else {
        $workflowToLoad.set({
          workflowId,
          mode,
          isOpen: true,
        });
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
    >
      <Flex flexDir="column" gap={2}>
        <Text>{t('nodes.loadWorkflowDesc')}</Text>
        <Text variant="subtext">{t('nodes.loadWorkflowDesc2')}</Text>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

LoadWorkflowConfirmationAlertDialog.displayName = 'LoadWorkflowConfirmationAlertDialog';
