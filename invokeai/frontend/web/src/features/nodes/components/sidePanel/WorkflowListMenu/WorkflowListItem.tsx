import { Badge, ConfirmationAlertDialog, Flex, IconButton, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { EMPTY_OBJECT } from 'app/store/constants';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useDisclosure } from 'common/hooks/useBoolean';
import dateFormat, { masks } from 'dateformat';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { selectWorkflowId, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useDeleteLibraryWorkflow } from 'features/workflowLibrary/hooks/useDeleteLibraryWorkflow';
import { useDownloadWorkflow } from 'features/workflowLibrary/hooks/useDownloadWorkflow';
import { useGetAndLoadLibraryWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadLibraryWorkflow';
import type { MouseEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiPencilBold, PiShareFatBold, PiTrashBold } from 'react-icons/pi';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

import { CopyWorkflowLinkModal, useCopyWorkflowLinkModal } from './CopyWorkflowLinkModal';
import { WorkflowListItemTooltip } from './WorkflowListItemTooltip';

export const WorkflowListItem = ({ workflow }: { workflow: WorkflowRecordListItemDTO }) => {
  const deleteConfirmationDialog = useDisclosure(false);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const projectUrl = useStore($projectUrl);
  const workflowListMenu = useWorkflowListMenu();
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const workflowId = useAppSelector(selectWorkflowId);
  const downloadWorkflow = useDownloadWorkflow();

  const copyWorkflowLinkModal = useCopyWorkflowLinkModal();

  const { deleteWorkflow, deleteWorkflowResult } = useDeleteLibraryWorkflow(EMPTY_OBJECT);
  const { getAndLoadWorkflow } = useGetAndLoadLibraryWorkflow({
    onSuccess: workflowListMenu.close,
  });

  const isActive = useMemo(() => {
    return workflowId === workflow.workflow_id;
  }, [workflowId, workflow.workflow_id]);

  const handleClickLoad = useCallback(() => {
    getAndLoadWorkflow(workflow.workflow_id);
    workflowListMenu.close();
  }, [getAndLoadWorkflow, workflow.workflow_id, workflowListMenu]);

  const handleClickEdit = useCallback(async () => {
    await getAndLoadWorkflow(workflow.workflow_id);
    dispatch(workflowModeChanged('edit'));
    workflowListMenu.close();
  }, [getAndLoadWorkflow, workflow.workflow_id, dispatch, workflowListMenu]);

  const handleDeleteWorklow = useCallback(() => {
    deleteWorkflow(workflow.workflow_id);
    deleteConfirmationDialog.close();
  }, [deleteWorkflow, workflow.workflow_id, deleteConfirmationDialog]);

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      deleteConfirmationDialog.open();
    },
    [deleteConfirmationDialog]
  );

  const handleClickShare = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      copyWorkflowLinkModal.open();
    },
    [copyWorkflowLinkModal]
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
        <Tooltip label={<WorkflowListItemTooltip workflow={workflow} />}>
          <Flex flexDir="column" gap={1}>
            <Flex gap={4} alignItems="center">
              <Text noOfLines={2}>{workflow.name}</Text>

              {isActive && (
                <Badge
                  color="invokeBlue.400"
                  borderColor="invokeBlue.700"
                  borderWidth={1}
                  bg="transparent"
                  flexShrink={0}
                >
                  {t('workflows.opened')}
                </Badge>
              )}
            </Flex>
            {workflow.category !== 'default' && (
              <Text fontSize="xs" variant="subtext" flexShrink={0} noOfLines={1}>
                {t('common.updated')}: {dateFormat(workflow.updated_at, masks.shortDate)}{' '}
                {dateFormat(workflow.updated_at, masks.shortTime)}
              </Text>
            )}
          </Flex>
        </Tooltip>
        <Spacer />

        <Flex alignItems="center" gap={1} opacity={isHovered ? 1 : 0}>
          <Tooltip label={t('workflows.edit')}>
            <IconButton
              size="sm"
              variant="ghost"
              aria-label={t('workflows.edit')}
              onClick={handleClickEdit}
              isLoading={deleteWorkflowResult.isLoading}
              icon={<PiPencilBold />}
            />
          </Tooltip>
          <Tooltip label={t('workflows.download')}>
            <IconButton
              size="sm"
              variant="ghost"
              aria-label={t('workflows.download')}
              onClick={downloadWorkflow}
              icon={<PiDownloadSimpleBold />}
            />
          </Tooltip>
          {!!projectUrl && workflow.workflow_id && (
            <Tooltip label={t('workflows.copyShareLink')}>
              <IconButton
                size="sm"
                variant="ghost"
                aria-label={t('workflows.copyShareLink')}
                onClick={handleClickShare}
                icon={<PiShareFatBold />}
              />
            </Tooltip>
          )}
          {workflow.category !== 'default' && (
            <Tooltip label={t('workflows.delete')}>
              <IconButton
                size="sm"
                variant="ghost"
                aria-label={t('workflows.delete')}
                onClick={handleClickDelete}
                isLoading={deleteWorkflowResult.isLoading}
                colorScheme="error"
                icon={<PiTrashBold />}
              />
            </Tooltip>
          )}
        </Flex>
      </Flex>
      <ConfirmationAlertDialog
        isOpen={deleteConfirmationDialog.isOpen}
        onClose={deleteConfirmationDialog.close}
        title={t('workflows.deleteWorkflow')}
        acceptCallback={handleDeleteWorklow}
        acceptButtonText={t('common.delete')}
        cancelButtonText={t('common.cancel')}
        useInert={false}
      >
        <p>{t('workflows.deleteWorkflow2')}</p>
      </ConfirmationAlertDialog>
      <CopyWorkflowLinkModal workflowId={workflow.workflow_id} workflowName={workflow.name} />
    </>
  );
};
