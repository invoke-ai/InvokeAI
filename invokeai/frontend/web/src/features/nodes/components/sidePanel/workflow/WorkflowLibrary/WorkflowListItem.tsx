import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Flex, Icon, Image, Spacer, Switch, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { selectWorkflowId } from 'features/nodes/store/selectors';
import { workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import InvokeLogo from 'public/assets/images/invoke-symbol-wht-lrg.svg';
import { type ChangeEvent, memo, type MouseEvent, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImage } from 'react-icons/pi';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useUpdateWorkflowIsPublicMutation } from 'services/api/endpoints/workflows';
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
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const isActive = useMemo(() => {
    return workflowId === workflow.workflow_id;
  }, [workflowId, workflow.workflow_id]);

  const isOwner = useMemo(() => {
    return currentUser !== null && workflow.user_id === currentUser.user_id;
  }, [currentUser, workflow.user_id]);

  const canEditOrDelete = useMemo(() => {
    // In single-user (legacy) mode, all workflows are editable — no concept of ownership.
    if (!setupStatus?.multiuser_enabled) {
      return true;
    }
    return isOwner || (currentUser?.is_admin ?? false);
  }, [setupStatus?.multiuser_enabled, isOwner, currentUser]);

  const tags = useMemo(() => {
    if (!workflow.tags) {
      return [];
    }
    return workflow.tags
      .split(',')
      .map((tag) => tag.trim())
      .filter((tag) => tag.length > 0);
  }, [workflow.tags]);

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
              {setupStatus?.multiuser_enabled && workflow.is_public && workflow.category !== 'default' && (
                <Badge
                  color="invokeGreen.400"
                  borderColor="invokeGreen.700"
                  borderWidth={1}
                  bg="transparent"
                  flexShrink={0}
                  variant="subtle"
                >
                  {t('workflows.shared')}
                </Badge>
              )}
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
          {tags.length > 0 && (
            <Text fontSize="xs" noOfLines={1}>
              <Text as="span" color="base.400">
                {t('workflows.tags')}:{' '}
              </Text>
              <Text as="span" color="base.400">
                {tags.join(', ')}
              </Text>
            </Text>
          )}
        </Flex>
        <Flex className={WORKFLOW_ACTION_BUTTONS_CN} alignItems="center" display="none" h={8}>
          {workflow.opened_at && (
            <Text variant="subtext" fontSize="xs" noOfLines={1} justifySelf="flex-end" pb={0.5}>
              {t('workflows.opened')} {new Date(workflow.opened_at).toLocaleString()}
            </Text>
          )}
          <Spacer />
          {setupStatus?.multiuser_enabled && canEditOrDelete && <ShareWorkflowToggle workflow={workflow} />}
          {workflow.category === 'default' && <ViewWorkflow workflowId={workflow.workflow_id} />}
          {workflow.category !== 'default' && (
            <>
              {canEditOrDelete && <EditWorkflow workflowId={workflow.workflow_id} />}
              <DownloadWorkflow workflowId={workflow.workflow_id} />
              {canEditOrDelete && <DeleteWorkflow workflowId={workflow.workflow_id} />}
            </>
          )}
        </Flex>
      </Flex>
    </Flex>
  );
});
WorkflowListItem.displayName = 'WorkflowListItem';

const ShareWorkflowToggle = memo(({ workflow }: { workflow: WorkflowRecordListItemWithThumbnailDTO }) => {
  const { t } = useTranslation();
  const [updateIsPublic, { isLoading }] = useUpdateWorkflowIsPublicMutation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      e.stopPropagation();
      updateIsPublic({ workflow_id: workflow.workflow_id, is_public: e.target.checked });
    },
    [updateIsPublic, workflow.workflow_id]
  );

  const handleClick = useCallback((e: MouseEvent) => {
    e.stopPropagation();
  }, []);

  return (
    <Tooltip label={t('workflows.shareWorkflow')}>
      <Flex alignItems="center" gap={1} onClick={handleClick}>
        <Text variant="subtext" fontSize="xs">
          {t('workflows.shared')}
        </Text>
        <Switch size="sm" isChecked={workflow.is_public} onChange={handleChange} isDisabled={isLoading} />
      </Flex>
    </Tooltip>
  );
});
ShareWorkflowToggle.displayName = 'ShareWorkflowToggle';

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
