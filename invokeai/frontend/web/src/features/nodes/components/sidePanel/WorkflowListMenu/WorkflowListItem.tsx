import { Badge, Flex, IconButton, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { useAppSelector } from 'app/store/storeHooks';
import dateFormat, { masks } from 'dateformat';
import { selectWorkflowId } from 'features/nodes/store/workflowSlice';
import { useDeleteWorkflow } from 'features/workflowLibrary/components/DeleteLibraryWorkflowConfirmationAlertDialog';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useDownloadWorkflow } from 'features/workflowLibrary/hooks/useDownloadWorkflow';
import type { MouseEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiPencilBold, PiShareFatBold, PiTrashBold } from 'react-icons/pi';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

import { useShareWorkflow } from './ShareWorkflowModal';
import { WorkflowListItemTooltip } from './WorkflowListItemTooltip';

export const WorkflowListItem = ({ workflow }: { workflow: WorkflowRecordListItemDTO }) => {
  const { t } = useTranslation();
  const projectUrl = useStore($projectUrl);
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const workflowId = useAppSelector(selectWorkflowId);
  const downloadWorkflow = useDownloadWorkflow();
  const shareWorkflow = useShareWorkflow();
  const deleteWorkflow = useDeleteWorkflow();
  const loadWorkflow = useLoadWorkflow();

  const isActive = useMemo(() => {
    return workflowId === workflow.workflow_id;
  }, [workflowId, workflow.workflow_id]);

  const handleClickLoad = useCallback(() => {
    setIsHovered(false);
    loadWorkflow.loadWithDialog(workflow.workflow_id, 'view');
  }, [loadWorkflow, workflow.workflow_id]);

  const handleClickEdit = useCallback(() => {
    setIsHovered(false);
    loadWorkflow.loadWithDialog(workflow.workflow_id, 'view');
  }, [loadWorkflow, workflow.workflow_id]);

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      deleteWorkflow(workflow);
    },
    [deleteWorkflow, workflow]
  );

  const handleClickShare = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      shareWorkflow(workflow);
    },
    [shareWorkflow, workflow]
  );

  const handleClickDownload = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      downloadWorkflow();
    },
    [downloadWorkflow]
  );

  return (
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
      <Tooltip label={<WorkflowListItemTooltip workflow={workflow} />} closeOnScroll>
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
        <Tooltip
          label={t('workflows.edit')}
          // This prevents an issue where the tooltip isn't closed after the modal is opened
          isOpen={!isHovered ? false : undefined}
          closeOnScroll
        >
          <IconButton
            size="sm"
            variant="ghost"
            aria-label={t('workflows.edit')}
            onClick={handleClickEdit}
            icon={<PiPencilBold />}
          />
        </Tooltip>
        <Tooltip
          label={t('workflows.download')}
          // This prevents an issue where the tooltip isn't closed after the modal is opened
          isOpen={!isHovered ? false : undefined}
          closeOnScroll
        >
          <IconButton
            size="sm"
            variant="ghost"
            aria-label={t('workflows.download')}
            onClick={handleClickDownload}
            icon={<PiDownloadSimpleBold />}
          />
        </Tooltip>
        {!!projectUrl && workflow.workflow_id && workflow.category !== 'user' && (
          <Tooltip
            label={t('workflows.copyShareLink')}
            // This prevents an issue where the tooltip isn't closed after the modal is opened
            isOpen={!isHovered ? false : undefined}
            closeOnScroll
          >
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
          <Tooltip
            label={t('workflows.delete')}
            // This prevents an issue where the tooltip isn't closed after the modal is opened
            isOpen={!isHovered ? false : undefined}
            closeOnScroll
          >
            <IconButton
              size="sm"
              variant="ghost"
              aria-label={t('workflows.delete')}
              onClick={handleClickDelete}
              colorScheme="error"
              icon={<PiTrashBold />}
            />
          </Tooltip>
        )}
      </Flex>
    </Flex>
  );
};
