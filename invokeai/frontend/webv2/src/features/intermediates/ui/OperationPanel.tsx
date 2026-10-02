import type { IntermediatesOperation } from '@features/intermediates/core/types';
import type { TFunction } from 'i18next';
/* eslint-disable react-perf/jsx-no-new-function-as-prop */

import { Alert, HStack, Progress, Stack, Text } from '@chakra-ui/react';
import {
  canRunOperationAgain,
  getOperationProcessed,
  getOperationTotalTargets,
  isOperationSettled,
} from '@features/intermediates/core/types';
import { formatBytes, formatCount } from '@platform/i18n/languages';
import { Button } from '@platform/ui/Button';
import { useTranslation } from 'react-i18next';

export interface OperationPanelProps {
  operation: IntermediatesOperation | null;
  isLoading: boolean;
  isRefetching: boolean;
  lookupError: string | null;
  /** False once the server has said it no longer knows the operation: asking again cannot help. */
  isLookupRetryable: boolean;
  onRefetch: () => void;
  /** Requests the operation's scope again; the button that asked is the trigger focus returns to. */
  onRunAgain: (trigger: HTMLElement) => void;
  onDismiss: () => void;
}

const Stat = ({ label, value }: { label: string; value: string }) => (
  <Stack gap="0" minW="4rem">
    <Text fontSize="xs" fontVariantNumeric="tabular-nums" fontWeight="600" lineHeight="shorter">
      {value}
    </Text>
    <Text color="fg.muted" fontSize="2xs" lineHeight="shorter">
      {label}
    </Text>
  </Stack>
);

const ANNOUNCED_PROGRESS_STEP = 25;

/**
 * The sentence screen readers hear. It changes with the status and at most every quarter of the progress, so a 2s poll
 * or a socket update that only moves the counters stays silent; the progress bar exposes the exact value on demand.
 */
export const getOperationAnnouncement = (
  operation: IntermediatesOperation | null,
  isLoading: boolean,
  t: TFunction,
  lookupError: string | null = null
): string => {
  if (lookupError) {
    return lookupError;
  }
  if (!operation) {
    return isLoading ? t('intermediates.operation.pending') : '';
  }
  const statusLabel = t(`intermediates.operation.status.${operation.status}`);
  if (isOperationSettled(operation) || operation.status === 'pending') {
    return statusLabel;
  }
  const steps = Math.floor(
    (getOperationProcessed(operation) / Math.max(getOperationTotalTargets(operation), 1)) *
      (100 / ANNOUNCED_PROGRESS_STEP)
  );
  return t('intermediates.operation.progressAnnouncement', {
    percent: (steps * ANNOUNCED_PROGRESS_STEP) / 100,
    status: statusLabel,
  });
};

/** The active or most recent cleanup: progress while it runs, a breakdown once it stops, another run for what failed. */
export const OperationPanel = ({
  isLoading,
  isLookupRetryable,
  isRefetching,
  lookupError,
  onRefetch,
  onDismiss,
  onRunAgain,
  operation,
}: OperationPanelProps) => {
  const { t } = useTranslation();

  if (lookupError) {
    return (
      <Alert.Root size="sm" status="error" variant="surface">
        <Alert.Indicator />
        <Alert.Content>
          <Alert.Title>{t('intermediates.operation.lookupFailed')}</Alert.Title>
          <Alert.Description>{lookupError}</Alert.Description>
          <HStack gap="2" mt="2">
            {isLookupRetryable ? (
              <Button loading={isRefetching} size="2xs" variant="outline" onClick={onRefetch}>
                {t('common.retry')}
              </Button>
            ) : null}
            <Button size="2xs" variant="ghost" onClick={onDismiss}>
              {t('intermediates.operation.dismiss')}
            </Button>
          </HStack>
        </Alert.Content>
      </Alert.Root>
    );
  }

  if (!operation) {
    return isLoading ? (
      <Text color="fg.muted" fontSize="xs">
        {t('intermediates.operation.pending')}
      </Text>
    ) : null;
  }

  const total = getOperationTotalTargets(operation);
  const processed = getOperationProcessed(operation);
  const settled = isOperationSettled(operation);
  const { progress } = operation;
  const statusLabel = t(`intermediates.operation.status.${operation.status}`);
  const failed = progress.failedImages + progress.failedVideos;

  return (
    <Alert.Root
      size="sm"
      status={operation.status === 'failed' ? 'error' : operation.status === 'completed' ? 'success' : 'info'}
      variant="surface"
    >
      <Alert.Indicator />
      <Alert.Content>
        <Alert.Title>{statusLabel}</Alert.Title>
        <Alert.Description asChild>
          <Stack gap="2" mt="1">
            {settled ? null : (
              <Stack gap="1">
                <Progress.Root
                  aria-label={statusLabel}
                  colorPalette="accent"
                  max={Math.max(total, 1)}
                  size="xs"
                  value={processed}
                >
                  <Progress.Track>
                    <Progress.Range />
                  </Progress.Track>
                </Progress.Root>
                <Text color="fg.muted" fontSize="2xs">
                  {t('intermediates.operation.progress', { done: formatCount(processed), total: formatCount(total) })}
                </Text>
              </Stack>
            )}
            {operation.error ? (
              <Text fontSize="xs" fontWeight="600">
                {operation.error}
              </Text>
            ) : null}
            <HStack gap="5" wrap="wrap">
              <Stat
                label={t('intermediates.operation.deleted')}
                value={formatCount(progress.deletedImages + progress.deletedVideos)}
              />
              <Stat
                label={t('intermediates.operation.retained')}
                value={formatCount(progress.retainedImages + progress.retainedVideos)}
              />
              <Stat label={t('intermediates.operation.failed')} value={formatCount(failed)} />
              {progress.pendingDiskCleanup > 0 ? (
                <Stat
                  label={t('intermediates.operation.pendingDisk')}
                  value={formatCount(progress.pendingDiskCleanup)}
                />
              ) : null}
              <Stat label={t('intermediates.operation.reclaimed')} value={formatBytes(progress.reclaimedBytes)} />
            </HStack>
            {progress.pendingDiskCleanup > 0 ? (
              <Text color="fg.muted" fontSize="2xs">
                {t('intermediates.operation.pendingDiskNote')}
              </Text>
            ) : null}
            {settled ? (
              <HStack gap="2">
                {canRunOperationAgain(operation) ? (
                  <Button size="2xs" variant="outline" onClick={(event) => onRunAgain(event.currentTarget)}>
                    {t('intermediates.operation.runAgain')}
                  </Button>
                ) : null}
                <Button size="2xs" variant="ghost" onClick={onDismiss}>
                  {t('intermediates.operation.dismiss')}
                </Button>
              </HStack>
            ) : null}
          </Stack>
        </Alert.Description>
      </Alert.Content>
    </Alert.Root>
  );
};
