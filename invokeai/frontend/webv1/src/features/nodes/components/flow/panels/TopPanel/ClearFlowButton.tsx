import { ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

const ClearFlowButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();

  const handleNewWorkflow = useCallback(() => {
    dispatch(nodeEditorReset());

    toast({
      id: 'WORKFLOW_CLEARED',
      title: t('workflows.workflowCleared'),
      status: 'success',
    });

    onClose();
  }, [dispatch, onClose, t]);

  const onClick = useCallback(() => {
    if (doesWorkflowHaveUnsavedChanges) {
      handleNewWorkflow();
      return;
    }
    onOpen();
  }, [doesWorkflowHaveUnsavedChanges, handleNewWorkflow, onOpen]);

  return (
    <>
      <IconButton
        tooltip={t('nodes.clearWorkflow')}
        aria-label={t('nodes.clearWorkflow')}
        icon={<PiTrashSimpleFill />}
        onClick={onClick}
        pointerEvents="auto"
      />
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('nodes.clearWorkflow')}
        acceptCallback={handleNewWorkflow}
        useInert={false}
      >
        <Flex flexDir="column" gap={2}>
          <Text>{t('nodes.clearWorkflowDesc')}</Text>
          <Text variant="subtext">{t('nodes.clearWorkflowDesc2')}</Text>
        </Flex>
      </ConfirmationAlertDialog>
    </>
  );
};

export default memo(ClearFlowButton);
