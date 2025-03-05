import { Badge, Flex, Icon, Image, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowId } from 'features/nodes/store/workflowSlice';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import InvokeLogo from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold, PiUsersBold } from 'react-icons/pi';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

import { DeleteWorkflow } from './WorkflowLibraryListItemActions/DeleteWorkflow';
import { DownloadWorkflow } from './WorkflowLibraryListItemActions/DownloadWorkflow';
import { EditWorkflow } from './WorkflowLibraryListItemActions/EditWorkflow';
import { SaveWorkflow } from './WorkflowLibraryListItemActions/SaveWorkflow';
import { ViewWorkflow } from './WorkflowLibraryListItemActions/ViewWorkflow';

const IMAGE_THUMBNAIL_SIZE = '80px';
const FALLBACK_ICON_SIZE = '24px';

export const WorkflowListItem = ({ workflow }: { workflow: WorkflowRecordListItemWithThumbnailDTO }) => {
  const { t } = useTranslation();
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseOver = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setIsHovered(false);
  }, []);

  const workflowId = useAppSelector(selectWorkflowId);
  const loadWorkflow = useLoadWorkflow();

  const isActive = useMemo(() => {
    return workflowId === workflow.workflow_id;
  }, [workflowId, workflow.workflow_id]);

  const handleClickLoad = useCallback(() => {
    setIsHovered(false);
    loadWorkflow.loadWithDialog(workflow.workflow_id, 'view');
  }, [loadWorkflow, workflow.workflow_id]);

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
        src={workflow.thumbnail_url ?? undefined}
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
          {workflow.category === 'default' && (
            <Image src={InvokeLogo} alt="invoke-logo" w="14px" h="14px" minW="14px" minH="14px" userSelect="none" />
          )}
        </Flex>

        <Flex alignItems="center" gap={1} opacity={isHovered ? 1 : 0}>
          {workflow.category === 'default' && (
            <>
              {/* need to consider what is useful here and which icons show that. idea is to "try it out"/"view" or "clone for your own changes" */}
              <ViewWorkflow workflowId={workflow.workflow_id} />
              <SaveWorkflow workflowId={workflow.workflow_id} />
            </>
          )}
          {workflow.category !== 'default' && (
            <>
              <EditWorkflow workflowId={workflow.workflow_id} />
              <DownloadWorkflow workflowId={workflow.workflow_id} />
              <DeleteWorkflow workflowId={workflow.workflow_id} />
            </>
          )}
        </Flex>
      </Flex>
    </Flex>
  );
};
