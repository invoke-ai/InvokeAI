import { Badge, Button, Flex, Icon, IconButton, Image, Spacer, Text, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowId } from 'features/nodes/store/workflowSlice';
import { useDeleteWorkflow } from 'features/workflowLibrary/components/DeleteLibraryWorkflowConfirmationAlertDialog';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useDownloadWorkflowById } from 'features/workflowLibrary/hooks/useDownloadWorkflowById';
import type { MouseEvent } from 'react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiDownloadSimpleBold,
  PiImageBold,
  PiPencilBold,
  PiShareFatBold,
  PiTrashBold,
  PiUsersBold,
} from 'react-icons/pi';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

import { useShareWorkflow } from './ShareWorkflowModal';

const IMAGE_THUMBNAIL_SIZE = '80px';
const FALLBACK_ICON_SIZE = '24px';

export const WorkflowListItem = ({ workflow }: { workflow: WorkflowRecordListItemWithThumbnailDTO }) => {
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
  const downloadWorkflowById = useDownloadWorkflowById();
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
    loadWorkflow.loadWithDialog(workflow.workflow_id, 'edit');
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
      downloadWorkflowById.downloadWorkflow(workflow.workflow_id);
    },
    [downloadWorkflowById, workflow.workflow_id]
  );

  return (
    <Flex
      gap={4}
      onClick={handleClickLoad}
      cursor="pointer"
      bg="base.750"
      _hover={{ backgroundColor: 'base.700' }}
      p={2}
      ps={3}
      borderRadius="base"
      w="full"
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
      alignItems="stretch"
    >
      <Image
        src=""
        fallbackStrategy="beforeLoadOrError"
        fallback={
          <Flex
            height={IMAGE_THUMBNAIL_SIZE}
            minWidth={IMAGE_THUMBNAIL_SIZE}
            bg="base.650"
            borderRadius="base"
            alignItems="center"
            justifyContent="center"
          >
            <Icon color="base.500" as={PiImageBold} boxSize={FALLBACK_ICON_SIZE} />
          </Flex>
        }
        objectFit="cover"
        objectPosition="50% 50%"
        height={IMAGE_THUMBNAIL_SIZE}
        width={IMAGE_THUMBNAIL_SIZE}
        minHeight={IMAGE_THUMBNAIL_SIZE}
        minWidth={IMAGE_THUMBNAIL_SIZE}
        borderRadius="base"
      />
      <Flex flexDir="column" gap={1} justifyContent="flex-start">
        <Flex gap={4} alignItems="center">
          <Text noOfLines={2}>{workflow.name}</Text>

          {isActive && (
            <Badge color="invokeBlue.400" borderColor="invokeBlue.700" borderWidth={1} bg="transparent" flexShrink={0}>
              {t('workflows.opened')}
            </Badge>
          )}
        </Flex>
        <Text variant="subtext" fontSize="xs" noOfLines={2}>
          {workflow.description}
        </Text>
      </Flex>

      <Spacer />
      <Flex flexDir="column" gap={1} justifyContent="space-between">
        <Flex gap={1} justifyContent="flex-end" w="full" p={2}>
          {workflow.category === 'project' && <Icon as={PiUsersBold} color="base.200" />}
          {workflow.category === 'default' && <Icon as={PiUsersBold} color="base.200" />}
        </Flex>

        {workflow.category !== 'default' ? (
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
                isLoading={downloadWorkflowById.isLoading}
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
          </Flex>
        ) : (
          <Flex flexDir="column" alignItems="center" gap={1} opacity={isHovered ? 1 : 0}>
            <Button size="xs">Try it out</Button>
            <Button size="xs">Copy to account</Button>
          </Flex>
        )}
      </Flex>
    </Flex>
  );
};
