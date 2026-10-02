import { Progress, Stack, Text } from '@chakra-ui/react';
import { formatBytes } from '@platform/i18n/languages';
import { useTranslation } from 'react-i18next';

import type { InstallByteProgress, InstallRowStatus } from './queueModel';

import { describeEta } from './queueModel';

const BAR_STATUSES: ReadonlySet<InstallRowStatus> = new Set(['downloading', 'installing', 'queued', 'paused']);

export const formatEta = (t: (key: string, options?: Record<string, unknown>) => string, etaSeconds: number) => {
  const eta = describeEta(etaSeconds);

  return t(`models.eta.${eta.unit}`, { count: eta.count });
};

export const InstallProgressBar = ({
  label,
  progress,
  status,
}: {
  label: string;
  progress: InstallByteProgress;
  status: InstallRowStatus;
}) => {
  // Installing has no byte budget: sweep until the backend reports completion.
  const value = status === 'installing' ? null : status === 'queued' ? 0 : (progress.ratio ?? 0);

  return (
    <Progress.Root
      aria-label={label}
      colorPalette={status === 'paused' ? 'gray' : 'accent'}
      max={1}
      minW="0"
      size="xs"
      value={value}
      w="full"
    >
      <Progress.Track rounded="full">
        <Progress.Range rounded="full" transition="width var(--wb-motion-duration-fast) ease" />
      </Progress.Track>
    </Progress.Root>
  );
};

/** One caption per row status, joined with middle dots so the column reads as a single line. */
export const useInstallProgressCaption = ({
  compact = false,
  progress,
  queuePosition,
  status,
  typeLabel,
}: {
  /** Drop the transfer rate so the caption fits a one-line summary bar. */
  compact?: boolean;
  progress: InstallByteProgress;
  queuePosition: number | null;
  status: InstallRowStatus;
  typeLabel: string | null;
}): string => {
  const { t } = useTranslation();
  const hasTotal = progress.totalBytes > 0;
  const doneOfTotal = `${formatBytes(progress.bytes)} / ${formatBytes(progress.totalBytes)}`;
  const parts: (string | null)[] = [];

  switch (status) {
    case 'downloading':
      if (hasTotal) {
        parts.push(`${Math.round((progress.ratio ?? 0) * 100)}%`, doneOfTotal);
      } else {
        parts.push(t('models.statusDownloading'), formatBytes(progress.bytes));
      }

      parts.push(
        progress.bytesPerSecond !== null && !compact
          ? t('models.transferRate', { rate: formatBytes(progress.bytesPerSecond) })
          : null,
        progress.etaSeconds !== null ? formatEta(t, progress.etaSeconds) : null
      );
      break;
    case 'installing':
      parts.push(t('models.installing'), hasTotal ? formatBytes(progress.totalBytes) : null);
      break;
    case 'queued':
      parts.push(
        t('models.statusWaiting'),
        queuePosition !== null ? t('models.queuePosition', { position: queuePosition }) : null,
        hasTotal ? formatBytes(progress.totalBytes) : null
      );
      break;
    case 'paused':
      parts.push(t('models.statusPaused'), hasTotal ? doneOfTotal : null);
      break;
    case 'installed':
      parts.push(t('common.done'), typeLabel, hasTotal ? formatBytes(progress.totalBytes) : null);
      break;
    case 'unauthorized':
      parts.push(t('models.progressNotStarted'), hasTotal ? formatBytes(progress.totalBytes) : null);
      break;
    case 'failed':
      parts.push(t('models.progressStopped'), t('models.bytesCopied', { bytes: formatBytes(progress.bytes) }));
      break;
    case 'cancelled':
      parts.push(t('models.statusCancelled'), t('models.bytesCopied', { bytes: formatBytes(progress.bytes) }));
      break;
  }

  return parts.filter((part): part is string => part !== null && part !== '').join(' · ');
};

export const InstallProgressCell = ({
  progress,
  queuePosition,
  status,
  typeLabel,
}: {
  progress: InstallByteProgress;
  queuePosition: number | null;
  status: InstallRowStatus;
  typeLabel: string | null;
}) => {
  const { t } = useTranslation();
  const caption = useInstallProgressCaption({ progress, queuePosition, status, typeLabel });

  return (
    <Stack gap="1" minW="0">
      {BAR_STATUSES.has(status) ? (
        <InstallProgressBar label={t('models.downloadProgress')} progress={progress} status={status} />
      ) : null}
      <Text color="fg.muted" fontFamily="mono" fontSize="2xs" lineHeight="short" overflowWrap="anywhere">
        {caption}
      </Text>
    </Stack>
  );
};
