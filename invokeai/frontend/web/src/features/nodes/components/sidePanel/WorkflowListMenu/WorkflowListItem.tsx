import {
  Badge,
  ConfirmationAlertDialog,
  Flex,
  IconButton,
  Spacer,
  Text,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { EMPTY_OBJECT } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { $isWorkflowListMenuIsOpen } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowId, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useDeleteLibraryWorkflow } from 'features/workflowLibrary/hooks/useDeleteLibraryWorkflow';
import { useDownloadWorkflow } from 'features/workflowLibrary/hooks/useDownloadWorkflow';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import type { MouseEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiPencilBold, PiTrashBold } from 'react-icons/pi';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

export const WorkflowListItem = ({ workflow }: { workflow: WorkflowRecordListItemDTO }) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const [isHovered, setIsHovered] = useState(false);

  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const workflowId = useAppSelector(selectWorkflowId);
  const downloadWorkflow = useDownloadWorkflow();

  const { deleteWorkflow, deleteWorkflowResult } = useDeleteLibraryWorkflow(EMPTY_OBJECT);
  const { getAndLoadWorkflow } = useGetAndLoadLibraryWorkflow({
    onSuccess: () => $isWorkflowListMenuIsOpen.set(false),
  });

  const isActive = useMemo(() => {
    return workflowId === workflow.workflow_id;
  }, [workflowId, workflow.workflow_id]);

  const handleClickLoad = useCallback(() => {
    getAndLoadWorkflow(workflow.workflow_id);
    $isWorkflowListMenuIsOpen.set(false);
  }, [workflow.workflow_id, getAndLoadWorkflow]);

  const handleClickEdit = useCallback(async () => {
    await getAndLoadWorkflow(workflow.workflow_id);
    dispatch(workflowModeChanged('edit'));
    $isWorkflowListMenuIsOpen.set(false);
  }, [workflow.workflow_id, dispatch, getAndLoadWorkflow]);

  const handleDeleteWorklow = useCallback(() => {
    deleteWorkflow(workflow.workflow_id);
    onClose();
  }, [workflow.workflow_id, deleteWorkflow, onClose]);

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      onOpen();
    },
    [onOpen]
  );

  return (
    <>
      <Flex
        gap={4}
        onClick={handleClickLoad}
        cursor="pointer"
        _hover={{ backgroundColor: 'base.750' }}
        p={2}
        ps={3}
        borderRadius="base"
        w="full"
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
        alignItems="center"
      >
        <Tooltip label={workflow.description}>
          <Text noOfLines={2}>{workflow.name}</Text>
        </Tooltip>
        {isActive && (
          <Badge color="invokeBlue.400" borderColor="invokeBlue.700" borderWidth={1} bg="transparent" flexShrink={0}>
            {t('workflows.opened')}
          </Badge>
        )}
        <Spacer />

        <Flex alignItems="center" gap={1} opacity={isHovered ? 1 : 0}>
          <IconButton
            size="sm"
            variant="ghost"
            aria-label="Edit"
            onClick={handleClickEdit}
            isLoading={deleteWorkflowResult.isLoading}
            icon={<PiPencilBold />}
          />
          <IconButton
            size="sm"
            variant="ghost"
            aria-label="Download"
            onClick={downloadWorkflow}
            icon={<PiDownloadSimpleBold />}
          />
          {workflow.category !== 'default' && (
            <IconButton
              size="sm"
              variant="ghost"
              aria-label={t('stylePresets.deleteTemplate')}
              onClick={handleClickDelete}
              isLoading={deleteWorkflowResult.isLoading}
              colorScheme="error"
              icon={<PiTrashBold />}
            />
          )}
        </Flex>
      </Flex>
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('workflows.deleteWorkflow')}
        acceptCallback={handleDeleteWorklow}
        acceptButtonText={t('common.delete')}
        cancelButtonText={t('common.cancel')}
        useInert={false}
      >
        <p>{t('workflows.deleteWorkflow2')}</p>
      </ConfirmationAlertDialog>
    </>
  );
};
