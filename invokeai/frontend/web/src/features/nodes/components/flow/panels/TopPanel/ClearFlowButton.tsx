import { ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { selectWorkflowIsTouched } from 'features/nodes/store/workflowSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleFill } from 'react-icons/pi';

const ClearFlowButton = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const isTouched = useAppSelector(selectWorkflowIsTouched);

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
    if (!isTouched) {
      handleNewWorkflow();
      return;
    }
    onOpen();
  }, [handleNewWorkflow, isTouched, onOpen]);

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
