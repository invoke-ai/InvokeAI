import { ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

const NewWorkflowButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const isTouched = useAppSelector((s) => s.workflow.isTouched);

  const handleNewWorkflow = useCallback(() => {
    dispatch(nodeEditorReset());
    dispatch(workflowModeChanged('edit'));

    dispatch(
      addToast(
        makeToast({
          title: t('workflows.newWorkflowCreated'),
          status: 'success',
        })
      )
    );

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
        aria-label={t('nodes.newWorkflow')}
        tooltip={t('nodes.newWorkflow')}
        icon={<PiFilePlusBold />}
        onClick={onClick}
        pointerEvents="auto"
      />
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('nodes.newWorkflow')}
        acceptCallback={handleNewWorkflow}
      >
        <Flex flexDir="column" gap={2}>
          <Text>{t('nodes.newWorkflowDesc')}</Text>
          <Text variant="subtext">{t('nodes.newWorkflowDesc2')}</Text>
        </Flex>
      </ConfirmationAlertDialog>
    </>
  );
};

export default memo(NewWorkflowButton);
