import { Flex, useDisclosure } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvText } from 'common/components/InvText/wrapper';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi'

const NewWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const isTouched = useAppSelector((state) => state.workflow.isTouched);

  const handleNewWorkflow = useCallback(() => {
    dispatch(nodeEditorReset());

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
      <InvMenuItem as="button" icon={<PiFlowArrowBold />} onClick={onClick}>
        {t('nodes.newWorkflow')}
      </InvMenuItem>

      <InvConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('nodes.newWorkflow')}
        acceptCallback={handleNewWorkflow}
      >
        <Flex flexDir="column" gap={2}>
          <InvText>{t('nodes.newWorkflowDesc')}</InvText>
          <InvText variant="subtext">{t('nodes.newWorkflowDesc2')}</InvText>
        </Flex>
      </InvConfirmationAlertDialog>
    </>
  );
};

export default memo(NewWorkflowMenuItem);
