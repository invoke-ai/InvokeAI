/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { ModelInstallJob } from '@workbench/models/types';

import { Badge, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { useConnectionStatusSelector } from '@workbench/backend/connectionStore';
import { IconButton, Tooltip } from '@workbench/components/ui';
import {
  cancelModelInstall,
  pauseModelInstall,
  restartFailedModelInstall,
  restartModelInstallFile,
  resumeModelInstall,
} from '@workbench/models/api';
import { isActiveInstallStatus, refreshInstalls, replaceInstallJob } from '@workbench/models/installsStore';
import { getInstallSourceLabel } from '@workbench/models/taxonomy';
import { PauseIcon, PlayIcon, RotateCcwIcon, TriangleAlertIcon, XIcon } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { InstallJobProgress } from './InstallJobProgress';
import { STATUS_BADGES, getInstallJobDisplayName } from './queueUtils';

const partFileName = (part: { url?: string; source?: string; local_path?: string }): string =>
  (part.url ?? part.source ?? part.local_path ?? 'file').split(/[\\/]/).at(-1) ?? 'file';

export const InstallJobRow = ({
  job,
  onError,
}: {
  job: ModelInstallJob;
  onError: (title: string, message: string) => void;
}) => {
  const { t } = useTranslation();
  const [isActing, setIsActing] = useState(false);
  const connectionStatus = useConnectionStatusSelector((snapshot) => snapshot.status);
  const badge = STATUS_BADGES[job.status] ?? { label: job.status, palette: 'gray' };
  const sourceLabel = getInstallSourceLabel(job.source);
  const displayName = getInstallJobDisplayName(job);
  // Parts the backend could not resume (or that errored) need a manual restart.
  const problemParts = (job.download_parts ?? []).filter(
    (part) => part.resume_required === true || part.status === 'error'
  );
  const showDisconnected = connectionStatus !== 'connected' && isActiveInstallStatus(job.status);

  const runAction = async (action: () => Promise<unknown>, failureTitle: string) => {
    setIsActing(true);

    try {
      const result = await action();

      if (result && typeof result === 'object' && 'id' in (result as ModelInstallJob)) {
        replaceInstallJob(result as ModelInstallJob);
      } else {
        await refreshInstalls();
      }
    } catch (actionError) {
      onError(failureTitle, actionError instanceof Error ? actionError.message : String(actionError));
    } finally {
      setIsActing(false);
    }
  };

  return (
    <Stack
      bg="bg"
      borderColor={job.status === 'error' ? 'border.error' : 'border.subtle'}
      borderWidth="1px"
      gap="1.5"
      p="2"
      rounded="md"
    >
      <HStack gap="2" justify="space-between">
        <Text flex="1" fontSize="2xs" fontWeight="600" minW="0" truncate title={sourceLabel}>
          {displayName}
        </Text>
        {showDisconnected ? (
          <Tooltip content={t('models.backendDisconnectedProgressStale')}>
            <Icon as={TriangleAlertIcon} boxSize="3" color="fg.warning" flexShrink={0} />
          </Tooltip>
        ) : null}
        <Badge colorPalette={badge.palette} flexShrink={0} fontSize="2xs" size="sm" variant="surface">
          {badge.label}
        </Badge>
        <HStack flexShrink={0} gap="0.5">
          {job.status === 'downloading' ? (
            <Tooltip content={t('models.pauseDownload')}>
              <IconButton
                aria-label={t('models.pauseInstall')}
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => pauseModelInstall(job.id), t('models.pauseFailed'))}
              >
                <Icon as={PauseIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          {job.status === 'paused' ? (
            <Tooltip content={t('models.resumeDownload')}>
              <IconButton
                aria-label={t('models.resumeInstall')}
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => resumeModelInstall(job.id), t('models.resumeFailed'))}
              >
                <Icon as={PlayIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          {job.status === 'error' ? (
            <Tooltip content={t('models.retryFailedDownload')}>
              <IconButton
                aria-label={t('models.retryInstall')}
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => restartFailedModelInstall(job.id), t('models.retryFailed'))}
              >
                <Icon as={RotateCcwIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
          {isActiveInstallStatus(job.status) || job.status === 'paused' ? (
            <Tooltip content={t('models.cancelInstall')}>
              <IconButton
                aria-label={t('models.cancelInstall')}
                disabled={isActing}
                size="2xs"
                variant="ghost"
                onClick={() => void runAction(() => cancelModelInstall(job.id), t('models.cancelFailed'))}
              >
                <Icon as={XIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          ) : null}
        </HStack>
      </HStack>
      {displayName !== sourceLabel ? (
        <Text color="fg.subtle" fontSize="2xs" truncate>
          {sourceLabel}
        </Text>
      ) : null}
      {job.status === 'downloading' || job.status === 'waiting' ? <InstallJobProgress job={job} /> : null}
      {job.status === 'error' && job.error ? (
        <Text color="fg.error" fontSize="2xs" overflowWrap="anywhere">
          {job.error_reason ? `${job.error_reason}: ` : ''}
          {job.error}
        </Text>
      ) : null}
      {problemParts.map((part) => {
        const isResume = part.resume_required === true;
        const partUrl = typeof part.url === 'string' ? part.url : undefined;

        return (
          <Stack key={part.url ?? part.source ?? partFileName(part)} bg="bg.muted" gap="0.5" p="1.5" rounded="sm">
            <HStack gap="1.5">
              <Icon as={TriangleAlertIcon} boxSize="3" color={isResume ? 'fg.warning' : 'fg.error'} flexShrink={0} />
              <Text flex="1" fontSize="2xs" minW="0" title={partFileName(part)} truncate>
                {partFileName(part)}
              </Text>
              <Badge
                colorPalette={isResume ? 'orange' : 'red'}
                flexShrink={0}
                fontSize="2xs"
                size="sm"
                variant="surface"
              >
                {isResume ? t('models.resumeRequired') : t('common.failed')}
              </Badge>
              {partUrl ? (
                <Tooltip content={t('models.restartThisFile')}>
                  <IconButton
                    aria-label={t('models.restartFile')}
                    disabled={isActing}
                    size="2xs"
                    variant="ghost"
                    onClick={() =>
                      void runAction(() => restartModelInstallFile(job.id, partUrl), t('models.restartFileFailed'))
                    }
                  >
                    <Icon as={RotateCcwIcon} boxSize="3" />
                  </IconButton>
                </Tooltip>
              ) : null}
            </HStack>
            {part.resume_message ? (
              <Text color="fg.subtle" fontSize="2xs" overflowWrap="anywhere">
                {part.resume_message}
              </Text>
            ) : null}
          </Stack>
        );
      })}
    </Stack>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
