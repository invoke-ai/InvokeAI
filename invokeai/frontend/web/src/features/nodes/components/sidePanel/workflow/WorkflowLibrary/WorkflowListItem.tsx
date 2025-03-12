import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Flex, Icon, Image, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ShareWorkflowButton } from 'features/nodes/components/sidePanel/workflow/WorkflowLibrary/WorkflowLibraryListItemActions/ShareWorkflow';
import { selectWorkflowId, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import InvokeLogo from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImage, PiUsersBold } from 'react-icons/pi';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

import { DeleteWorkflow } from './WorkflowLibraryListItemActions/DeleteWorkflow';
import { DownloadWorkflow } from './WorkflowLibraryListItemActions/DownloadWorkflow';
import { EditWorkflow } from './WorkflowLibraryListItemActions/EditWorkflow';
import { ViewWorkflow } from './WorkflowLibraryListItemActions/ViewWorkflow';

const IMAGE_THUMBNAIL_SIZE = '108px';
const FALLBACK_ICON_SIZE = '32px';

const WORKFLOW_ACTION_BUTTONS_CN = 'workflow-action-buttons';

const sx: SystemStyleObject = {
  _hover: {
    bg: 'base.700',
    [`& .${WORKFLOW_ACTION_BUTTONS_CN}`]: {
      display: 'flex',
    },
  },
};

export const WorkflowListItem = memo(({ workflow }: { workflow: WorkflowRecordListItemWithThumbnailDTO }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const workflowId = useAppSelector(selectWorkflowId);
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const isActive = useMemo(() => {
    return workflowId === workflow.workflow_id;
  }, [workflowId, workflow.workflow_id]);

  const handleClickLoad = useCallback(() => {
    loadWorkflowWithDialog({
      type: 'library',
      data: workflow.workflow_id,
      onSuccess: () => {
        dispatch(workflowModeChanged('view'));
      },
    });
  }, [dispatch, loadWorkflowWithDialog, workflow.workflow_id]);

  return (
    <Flex
      position="relative"
      role="button"
      onClick={handleClickLoad}
      cursor="pointer"
      bg="base.750"
      borderRadius="base"
      w="full"
      alignItems="stretch"
      sx={sx}
      gap={2}
    >
      <Flex p={2} pr={0}>
        <Image
          src={workflow.thumbnail_url ?? undefined}
          fallbackStrategy="beforeLoadOrError"
          fallback={workflow.category === 'default' ? <DefaultThumbnailFallback /> : <UserThumbnailFallback />}
          objectFit="cover"
          objectPosition="50% 50%"
          height={IMAGE_THUMBNAIL_SIZE}
          width={IMAGE_THUMBNAIL_SIZE}
          minHeight={IMAGE_THUMBNAIL_SIZE}
          minWidth={IMAGE_THUMBNAIL_SIZE}
          borderRadius="base"
        />
      </Flex>
      <Flex flexDir="column" gap={1} justifyContent="space-between" w="full">
        <Flex flexDir="column" gap={1} alignItems="flex-start" pt={2} pe={2} w="full">
          <Flex gap={2} alignItems="flex-start" justifyContent="space-between" w="full">
            <Text noOfLines={2}>{workflow.name}</Text>
            <Flex gap={2} alignItems="center">
              {isActive && (
                <Badge
                  color="invokeBlue.400"
                  borderColor="invokeBlue.700"
                  borderWidth={1}
                  bg="transparent"
                  flexShrink={0}
                  variant="subtle"
                >
                  {t('workflows.opened')}
                </Badge>
              )}
              {workflow.category === 'project' && <Icon as={PiUsersBold} color="base.200" />}
              {workflow.category === 'default' && (
                <Image
                  src={InvokeLogo}
                  alt="invoke-logo"
                  w="14px"
                  h="14px"
                  minW="14px"
                  minH="14px"
                  userSelect="none"
                  opacity={0.5}
                />
              )}
            </Flex>
          </Flex>
          <Text variant="subtext" fontSize="xs" noOfLines={3}>
            {workflow.description}
          </Text>
        </Flex>
        <Flex className={WORKFLOW_ACTION_BUTTONS_CN} alignItems="center" display="none" h={8}>
          {workflow.opened_at && (
            <Text variant="subtext" fontSize="xs" noOfLines={1} justifySelf="flex-end" pb={0.5}>
              {t('workflows.opened')} {new Date(workflow.opened_at).toLocaleString()}
            </Text>
          )}
          <Spacer />
          {workflow.category === 'default' && <ViewWorkflow workflowId={workflow.workflow_id} />}
          {workflow.category !== 'default' && (
            <>
              <EditWorkflow workflowId={workflow.workflow_id} />
              <DownloadWorkflow workflowId={workflow.workflow_id} />
              <DeleteWorkflow workflowId={workflow.workflow_id} />
            </>
          )}
          {workflow.category === 'project' && <ShareWorkflowButton workflow={workflow} />}
        </Flex>
      </Flex>
    </Flex>
  );
});
WorkflowListItem.displayName = 'WorkflowListItem';

const UserThumbnailFallback = memo(() => {
  return (
    <Flex
      height={IMAGE_THUMBNAIL_SIZE}
      minWidth={IMAGE_THUMBNAIL_SIZE}
      bg="base.600"
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
      opacity={0.3}
    >
      <Icon as={PiImage} boxSize={FALLBACK_ICON_SIZE} />
    </Flex>
  );
});
UserThumbnailFallback.displayName = 'UserThumbnailFallback';

const DefaultThumbnailFallback = memo(() => {
  return (
    <Flex
      height={IMAGE_THUMBNAIL_SIZE}
      minWidth={IMAGE_THUMBNAIL_SIZE}
      bg="base.600"
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
      opacity={0.3}
    >
      <Image src={InvokeLogo} alt="invoke-logo" userSelect="none" boxSize={FALLBACK_ICON_SIZE} p={1} />
    </Flex>
  );
});
DefaultThumbnailFallback.displayName = 'DefaultThumbnailFallback';
