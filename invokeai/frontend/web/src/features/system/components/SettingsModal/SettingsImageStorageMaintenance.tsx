import { Button, Flex, FormControl, FormLabel, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { api, LIST_TAG } from 'services/api';
import { useGetRuntimeConfigQuery } from 'services/api/endpoints/appInfo';
import {
  useGetImageMoveStatusQuery,
  useStartImageMoveMutation,
  useStartImageMoveRecoveryMutation,
} from 'services/api/endpoints/imageMoves';
import type { S } from 'services/api/types';

const getOperationKey = (operation: S['ImageMoveStatusResponse']['operation']) => {
  if (operation === 'move_all') {
    return 'settings.imageStorageMaintenanceOperationMove';
  }
  if (operation === 'recovery') {
    return 'settings.imageStorageMaintenanceOperationRecovery';
  }
  return 'settings.imageStorageMaintenanceOperationNone';
};

const getJobStateKey = (state: S['ImageMoveJobResponse']['state'] | undefined) => {
  if (state === 'planned') {
    return 'settings.imageStorageMaintenanceStatePlanned';
  }
  if (state === 'moving') {
    return 'settings.imageStorageMaintenanceStateMoving';
  }
  if (state === 'moved') {
    return 'settings.imageStorageMaintenanceStateMoved';
  }
  if (state === 'committed') {
    return 'settings.imageStorageMaintenanceStateCommitted';
  }
  if (state === 'error') {
    return 'settings.imageStorageMaintenanceStateError';
  }
  return 'settings.imageStorageMaintenanceStateNone';
};

const invalidatedImageMoveJobIds = new Set<number>();

export const SettingsImageStorageMaintenance = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: runtimeConfig } = useGetRuntimeConfigQuery();
  // The S3 backend has no filesystem move/reorganize path, so storage
  // maintenance (move-all / recovery) is not applicable and is hidden.
  const isS3Backend = runtimeConfig?.config.storage_backend === 's3';
  const canAccess = runtimeConfig
    ? (!runtimeConfig.config.multiuser || Boolean(currentUser?.is_admin)) && !isS3Backend
    : false;
  const [startImageMove, startImageMoveState] = useStartImageMoveMutation();
  const [startImageMoveRecovery, startImageMoveRecoveryState] = useStartImageMoveRecoveryMutation();
  const [shouldPollStatus, setShouldPollStatus] = useState(false);
  const { data: status, isFetching } = useGetImageMoveStatusQuery(undefined, {
    skip: !canAccess,
    pollingInterval: shouldPollStatus ? 2000 : 0,
  });

  const isRunning = status?.is_running ?? false;
  const latestJob = status?.latest_job;
  const needsMoveCount = status?.needs_move_count ?? 0;
  const hasActiveJob = status?.active_job_id !== null && status?.active_job_id !== undefined;
  const isBusy = isRunning || startImageMoveState.isLoading || startImageMoveRecoveryState.isLoading;

  useEffect(() => {
    setShouldPollStatus(canAccess && isRunning);
  }, [canAccess, isRunning]);

  useEffect(() => {
    if (!latestJob || latestJob.state !== 'committed' || isRunning) {
      return;
    }
    if (invalidatedImageMoveJobIds.has(latestJob.id)) {
      return;
    }
    invalidatedImageMoveJobIds.add(latestJob.id);
    dispatch(
      api.util.invalidateTags([
        'Image',
        'ImageList',
        'ImageMetadata',
        'ImageWorkflow',
        'ImageNameList',
        'ImageCollectionCounts',
        'ImageMoveStatus',
        { type: 'ImageCollection', id: LIST_TAG },
      ])
    );
  }, [dispatch, isRunning, latestJob]);

  const statusText = useMemo(() => {
    if (!status) {
      return t('settings.imageStorageMaintenanceStatusUnavailable');
    }
    if (hasActiveJob && !isRunning) {
      return t('settings.imageStorageMaintenanceStatusNeedsRecovery', { jobId: status.active_job_id });
    }
    if (isRunning) {
      return t('settings.imageStorageMaintenanceStatusRunning', { operation: t(getOperationKey(status.operation)) });
    }
    if (needsMoveCount > 0) {
      return t('settings.imageStorageMaintenanceStatusIncomplete', { count: needsMoveCount });
    }
    return t('settings.imageStorageMaintenanceStatusIdle', { state: t(getJobStateKey(latestJob?.state)) });
  }, [hasActiveJob, isRunning, latestJob?.state, needsMoveCount, status, t]);

  const onStart = useCallback(async () => {
    try {
      setShouldPollStatus(true);
      await startImageMove().unwrap();
    } catch {
      setShouldPollStatus(false);
      toast({
        id: 'IMAGE_STORAGE_MAINTENANCE_START_FAILED',
        title: t('settings.imageStorageMaintenanceStartFailed'),
        status: 'error',
      });
    }
  }, [startImageMove, t]);

  const onRecover = useCallback(async () => {
    try {
      setShouldPollStatus(true);
      await startImageMoveRecovery().unwrap();
    } catch {
      setShouldPollStatus(false);
      toast({
        id: 'IMAGE_STORAGE_MAINTENANCE_RECOVERY_FAILED',
        title: t('settings.imageStorageMaintenanceRecoveryFailed'),
        status: 'error',
      });
    }
  }, [startImageMoveRecovery, t]);

  if (!canAccess) {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('settings.imageStorageMaintenance')}</FormLabel>
      <Flex gap={2} alignItems="center" flexWrap="wrap">
        <Button
          onClick={onStart}
          isLoading={startImageMoveState.isLoading}
          isDisabled={isBusy || isFetching || hasActiveJob}
        >
          {t('settings.imageStorageMaintenanceStart')}
        </Button>
        <Button
          onClick={onRecover}
          isLoading={startImageMoveRecoveryState.isLoading}
          isDisabled={isBusy || isFetching}
          variant="outline"
        >
          {t('settings.imageStorageMaintenanceRecover')}
        </Button>
      </Flex>
      <Text variant="subtext">{statusText}</Text>
      {latestJob?.error_message || status?.last_error ? (
        <Text variant="subtext">{latestJob?.error_message || status?.last_error}</Text>
      ) : null}
    </FormControl>
  );
});

SettingsImageStorageMaintenance.displayName = 'SettingsImageStorageMaintenance';
