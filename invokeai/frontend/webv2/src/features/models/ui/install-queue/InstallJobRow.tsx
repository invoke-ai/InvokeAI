import { Box, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { useInstallProgress } from '@features/models/data/installsStore';
import { useModelsSelector } from '@features/models/data/modelsStore';
import { openModelDetail, openModelManagerTab } from '@features/models/ui/uiStore';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { Button } from '@platform/ui/Button';
import {
  ManagerQueueErrorDetail,
  ManagerQueueRow,
  ManagerQueueRowAction,
  type ManagerQueueGlyph,
} from '@platform/ui/ManagerQueue';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import {
  CheckIcon,
  CircleStopIcon,
  ClockIcon,
  LockIcon,
  MinusIcon,
  PauseIcon,
  PlayIcon,
  RotateCcwIcon,
  SquareArrowOutUpRightIcon,
  TriangleAlertIcon,
  XIcon,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { InstallQueueRow, InstallRowStatus } from './queueModel';

import { hasInstallProgressBar, InstallProgressBar, useInstallProgressCaption } from './InstallProgressCell';
import {
  canRestartFailedParts,
  canResumeInstall,
  getInstallJobDisplayName,
  getInstallJobSourceLabel,
  getProblemDownloadParts,
  isActiveRowStatus,
  isHuggingFaceSource,
  isSettledRowStatus,
  resolveInstallProgress,
  STATUS_PRESENTATION,
} from './queueModel';
import { useConfirmedInstallCancel } from './useConfirmedInstallCancel';
import { useInstallJobActions } from './useInstallJobActions';

const STATUS_GLYPHS: Record<InstallRowStatus, ManagerQueueGlyph> = {
  cancelled: { color: 'fg.muted', icon: MinusIcon },
  downloading: 'active',
  failed: { color: 'fg.error', icon: TriangleAlertIcon },
  installed: { color: 'fg.success', icon: CheckIcon },
  installing: 'active',
  paused: { color: 'fg.muted', icon: PauseIcon },
  queued: { color: 'fg.muted', icon: ClockIcon },
  unauthorized: { color: 'fg.warning', icon: LockIcon },
};

export const InstallJobRow = ({ row }: { row: InstallQueueRow }) => {
  const { job, queuePosition, status } = row;
  const { t } = useTranslation();
  const actions = useInstallJobActions(job);
  const liveProgress = useInstallProgress(job.id);
  const connectionStatus = useConnectionStatusSelector((snapshot) => snapshot.status);
  const installedKey = job.config_out?.key ?? null;
  const isInLibrary = useModelsSelector((snapshot) =>
    installedKey !== null ? snapshot.modelsByKey.has(installedKey) : false
  );

  const presentation = STATUS_PRESENTATION[status];
  const progress = resolveInstallProgress(job, liveProgress);
  const cancel = useConfirmedInstallCancel(job, progress, actions.cancel);
  const displayName = getInstallJobDisplayName(job);
  const sourceLabel = getInstallJobSourceLabel(job);
  const problemParts = getProblemDownloadParts(job);
  const typeLabel = job.config_out?.type ? getModelTypeLabel(job.config_out.type) : null;
  const caption = useInstallProgressCaption({ progress, queuePosition, status, typeLabel });
  const showDisconnected = connectionStatus !== 'connected' && isActiveRowStatus(status);
  const tone = status === 'failed' ? 'error' : status === 'unauthorized' ? 'warning' : undefined;

  return (
    <ManagerQueueRow
      caption={caption}
      data-install-status={status}
      glyph={STATUS_GLYPHS[status]}
      name={displayName}
      nameMuted={status === 'installed'}
      progressBar={
        hasInstallProgressBar(status) ? (
          <InstallProgressBar label={t('models.downloadProgress')} progress={progress} status={status} />
        ) : null
      }
      status={{ label: t(presentation.labelKey), palette: presentation.palette }}
      statusAdornment={
        showDisconnected ? (
          <Tooltip content={t('models.backendDisconnectedProgressStale')}>
            <Box display="inline-flex">
              <Icon as={TriangleAlertIcon} boxSize="3" color="fg.warning" />
            </Box>
          </Tooltip>
        ) : null
      }
      subtitle={sourceLabel !== displayName ? sourceLabel : null}
      tone={tone}
      details={
        <>
          {status === 'failed' && job.error ? (
            <ManagerQueueErrorDetail message={job.error} reason={job.error_reason} />
          ) : null}
          {status === 'unauthorized' ? (
            <HStack align="baseline" flexWrap="wrap" gap="1.5" pt="0.5">
              <Text color="fg" fontSize="xs">
                {t(isHuggingFaceSource(job) ? 'models.gatedRepoMessage' : 'models.accessTokenRequiredMessage')}
              </Text>
              <Button
                fontSize="xs"
                h="auto"
                minW="0"
                p="0"
                textDecoration="underline"
                variant="plain"
                onClick={() => openModelManagerTab('keys')}
              >
                {t('models.addToken')}
              </Button>
            </HStack>
          ) : null}
          {problemParts.map((part) => (
            <Stack key={part.key} gap="0" pt="0.5">
              <HStack gap="1.5">
                <Icon
                  as={TriangleAlertIcon}
                  boxSize="3"
                  color={part.resumeRequired ? 'fg.warning' : 'fg.error'}
                  flexShrink={0}
                />
                <MiddleTruncate color="fg" fontFamily="mono" fontSize="xs" minW="0" text={part.fileName} />
                <Text color="fg.muted" fontSize="xs" whiteSpace="nowrap">
                  {part.resumeRequired ? t('models.resumeRequired') : t('common.failed')}
                </Text>
                {part.url ? (
                  <ManagerQueueRowAction
                    disabled={actions.isBusy}
                    icon={RotateCcwIcon}
                    label={t('models.restartThisFile')}
                    onClick={() => actions.restartFile(part.url!)}
                  />
                ) : null}
              </HStack>
              {part.message ? (
                <Text color="fg" fontSize="xs" lineClamp={2} overflowWrap="anywhere" ps="4.5">
                  {part.message}
                </Text>
              ) : null}
            </Stack>
          ))}
        </>
      }
      actions={
        <>
          {status === 'downloading' ? (
            <ManagerQueueRowAction
              disabled={actions.isBusy}
              icon={PauseIcon}
              label={t('models.pauseDownload')}
              onClick={actions.pause}
            />
          ) : null}
          {canResumeInstall(job) ? (
            <ManagerQueueRowAction
              disabled={actions.isBusy}
              icon={PlayIcon}
              label={t('models.resumeDownload')}
              onClick={actions.resume}
            />
          ) : null}
          {status === 'paused' && canRestartFailedParts(job) ? (
            <ManagerQueueRowAction
              disabled={actions.isBusy}
              icon={RotateCcwIcon}
              label={t('models.retryFailedDownload')}
              onClick={actions.retry}
            />
          ) : null}
          {isSettledRowStatus(status) && status !== 'installed' ? (
            <ManagerQueueRowAction
              disabled={actions.isBusy}
              icon={RotateCcwIcon}
              label={t('models.retryInstall')}
              onClick={actions.retry}
            />
          ) : null}
          {status === 'installed' && isInLibrary && installedKey !== null ? (
            <ManagerQueueRowAction
              disabled={false}
              icon={SquareArrowOutUpRightIcon}
              label={t('models.openModel')}
              onClick={() => openModelDetail(installedKey)}
            />
          ) : null}
          {isSettledRowStatus(status) ? (
            <ManagerQueueRowAction
              disabled={false}
              icon={XIcon}
              label={t('models.removeFromList')}
              onClick={actions.dismiss}
            />
          ) : (
            // Stop, not the X that removes a settled row from the list: this one throws work away.
            <ManagerQueueRowAction
              disabled={actions.isBusy}
              icon={CircleStopIcon}
              label={t('models.cancelInstall')}
              tone="danger"
              onClick={cancel.requestCancel}
            />
          )}
          {cancel.dialog}
        </>
      }
    />
  );
};
