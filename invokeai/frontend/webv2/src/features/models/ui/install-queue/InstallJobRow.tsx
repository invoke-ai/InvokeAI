import type { LucideIcon } from 'lucide-react';

import { Badge, Box, HStack, Icon, Spinner, Stack, Table, Text } from '@chakra-ui/react';
import { getModelTypeLabel } from '@features/models/core/taxonomy';
import { useInstallProgress } from '@features/models/data/installsStore';
import { useModelsSelector } from '@features/models/data/modelsStore';
import { openModelDetail, openModelManagerTab } from '@features/models/ui/uiStore';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { Button, IconButton, Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import {
  CheckIcon,
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

import { InstallProgressCell } from './InstallProgressCell';
import {
  canRestartFailedParts,
  getInstallJobDisplayName,
  getInstallJobSourceLabel,
  getProblemDownloadParts,
  isActiveRowStatus,
  isHuggingFaceSource,
  isSettledRowStatus,
  resolveInstallProgress,
  STATUS_PRESENTATION,
} from './queueModel';
import { useInstallJobActions } from './useInstallJobActions';

const STATUS_ICONS: Record<
  Exclude<InstallRowStatus, 'downloading' | 'installing'>,
  { icon: LucideIcon; color: string }
> = {
  cancelled: { color: 'fg.muted', icon: MinusIcon },
  failed: { color: 'fg.error', icon: TriangleAlertIcon },
  installed: { color: 'fg.success', icon: CheckIcon },
  paused: { color: 'fg.muted', icon: PauseIcon },
  queued: { color: 'fg.muted', icon: ClockIcon },
  unauthorized: { color: 'fg.warning', icon: LockIcon },
};

const CELL_PROPS = { borderColor: 'border.subtle', py: '2', verticalAlign: 'top' } as const;

const StatusGlyph = ({ status }: { status: InstallRowStatus }) => {
  if (isActiveRowStatus(status)) {
    return <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" />;
  }

  const { color, icon } = STATUS_ICONS[status];

  return <Icon as={icon} boxSize="3.5" color={color} />;
};

const RowAction = ({
  disabled,
  icon,
  label,
  onClick,
  tone,
}: {
  disabled: boolean;
  icon: LucideIcon;
  label: string;
  onClick: () => void;
  tone?: 'danger';
}) => (
  <Tooltip content={label}>
    <IconButton
      aria-label={label}
      colorPalette={tone === 'danger' ? 'danger' : undefined}
      disabled={disabled}
      size="2xs"
      variant="ghost"
      onClick={onClick}
    >
      <Icon as={icon} boxSize="3" />
    </IconButton>
  </Tooltip>
);

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
  const displayName = getInstallJobDisplayName(job);
  const sourceLabel = getInstallJobSourceLabel(job);
  const problemParts = getProblemDownloadParts(job);
  const typeLabel = job.config_out?.type ? getModelTypeLabel(job.config_out.type) : null;
  const showDisconnected = connectionStatus !== 'connected' && isActiveRowStatus(status);
  const tone = status === 'failed' ? 'error' : status === 'unauthorized' ? 'warning' : undefined;

  return (
    <Table.Row data-install-status={status} data-tone={tone}>
      <Table.Cell {...CELL_PROPS} ps="3" pt="2.5">
        <StatusGlyph status={status} />
      </Table.Cell>

      <Table.Cell {...CELL_PROPS}>
        <Stack gap="0" minW="0">
          <MiddleTruncate
            color={status === 'installed' ? 'fg.muted' : 'fg'}
            fontSize="xs"
            fontWeight="600"
            lineHeight="short"
            text={displayName}
          />
          {sourceLabel !== displayName ? (
            <MiddleTruncate color="fg.muted" fontFamily="mono" fontSize="2xs" text={sourceLabel} />
          ) : null}
          {status === 'failed' && job.error ? (
            <HStack align="baseline" flexWrap="wrap" gap="1.5" pt="0.5">
              {job.error_reason ? (
                <Badge colorPalette="red" fontFamily="mono" size="sm" variant="surface">
                  {job.error_reason}
                </Badge>
              ) : null}
              <Text color="fg" fontSize="2xs" lineClamp={2} overflowWrap="anywhere">
                {job.error}
              </Text>
            </HStack>
          ) : null}
          {status === 'unauthorized' ? (
            <HStack align="baseline" flexWrap="wrap" gap="1.5" pt="0.5">
              <Text color="fg" fontSize="2xs">
                {t(isHuggingFaceSource(job) ? 'models.gatedRepoMessage' : 'models.accessTokenRequiredMessage')}
              </Text>
              <Button
                fontSize="2xs"
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
            <HStack key={part.key} gap="1.5" pt="0.5">
              <Icon
                as={TriangleAlertIcon}
                boxSize="3"
                color={part.resumeRequired ? 'fg.warning' : 'fg.error'}
                flexShrink={0}
              />
              <MiddleTruncate color="fg" fontFamily="mono" fontSize="2xs" minW="0" text={part.fileName} />
              <Text color="fg.muted" fontSize="2xs" whiteSpace="nowrap">
                {part.message ?? (part.resumeRequired ? t('models.resumeRequired') : t('common.failed'))}
              </Text>
              {part.url ? (
                <RowAction
                  disabled={actions.isBusy}
                  icon={RotateCcwIcon}
                  label={t('models.restartThisFile')}
                  onClick={() => actions.restartFile(part.url!)}
                />
              ) : null}
            </HStack>
          ))}
        </Stack>
      </Table.Cell>

      <Table.Cell {...CELL_PROPS}>
        <InstallProgressCell progress={progress} queuePosition={queuePosition} status={status} typeLabel={typeLabel} />
      </Table.Cell>

      <Table.Cell {...CELL_PROPS}>
        <HStack gap="1.5">
          <Badge
            colorPalette={presentation.palette}
            fontSize="2xs"
            fontWeight="700"
            letterSpacing="wider"
            size="sm"
            textTransform="uppercase"
            variant="surface"
          >
            {t(presentation.labelKey)}
          </Badge>
          {showDisconnected ? (
            <Tooltip content={t('models.backendDisconnectedProgressStale')}>
              <Box display="inline-flex">
                <Icon as={TriangleAlertIcon} boxSize="3" color="fg.warning" />
              </Box>
            </Tooltip>
          ) : null}
        </HStack>
      </Table.Cell>

      <Table.Cell {...CELL_PROPS} pe="2">
        <HStack gap="0.5" justify="flex-end">
          {status === 'downloading' ? (
            <RowAction
              disabled={actions.isBusy}
              icon={PauseIcon}
              label={t('models.pauseDownload')}
              onClick={actions.pause}
            />
          ) : null}
          {status === 'paused' ? (
            <RowAction
              disabled={actions.isBusy}
              icon={canRestartFailedParts(job) ? RotateCcwIcon : PlayIcon}
              label={canRestartFailedParts(job) ? t('models.retryFailedDownload') : t('models.resumeDownload')}
              onClick={canRestartFailedParts(job) ? actions.retry : actions.resume}
            />
          ) : null}
          {isSettledRowStatus(status) && status !== 'installed' ? (
            <RowAction
              disabled={actions.isBusy}
              icon={RotateCcwIcon}
              label={t('models.retryInstall')}
              onClick={actions.retry}
            />
          ) : null}
          {status === 'installed' && isInLibrary && installedKey !== null ? (
            <RowAction
              disabled={false}
              icon={SquareArrowOutUpRightIcon}
              label={t('models.openModel')}
              onClick={() => openModelDetail(installedKey)}
            />
          ) : null}
          {isSettledRowStatus(status) ? (
            <RowAction disabled={false} icon={XIcon} label={t('models.removeFromList')} onClick={actions.dismiss} />
          ) : (
            <RowAction
              disabled={actions.isBusy}
              icon={XIcon}
              label={t('models.cancelInstall')}
              tone="danger"
              onClick={actions.cancel}
            />
          )}
        </HStack>
      </Table.Cell>
    </Table.Row>
  );
};
