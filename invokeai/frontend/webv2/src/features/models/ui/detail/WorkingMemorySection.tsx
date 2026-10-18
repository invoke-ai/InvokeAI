/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type {
  CalibrationActivity,
  CalibrationBlocker,
  CalibrationOutcome,
  WorkingMemoryPlan,
  WorkingMemoryStatus,
  WorkingMemorySummary,
} from '@features/models/core/workingMemory';

import { Alert, HStack, Progress, Skeleton, Stack, Text } from '@chakra-ui/react';
import {
  deriveCalibrationProgress,
  describeCoverage,
  getCalibrationBlocker,
  isMeasurementMethodVerified,
  planCalibrationSteps,
} from '@features/models/core/workingMemory';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  attachWorkingMemoryRealtime,
  resetCalibration,
  startCalibration,
  stopCalibration,
  useLocalAttempt,
} from '@features/models/data/workingMemoryCalibration';
import {
  workingMemoryPlanQueryOptions,
  workingMemoryStatusQueryOptions,
  workingMemorySummaryQueryOptions,
} from '@features/models/data/workingMemoryQueries';
import { useCalibrationGraphCompiler } from '@features/models/ui/ModelsUiContext';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useExitPresence } from '@platform/react/useExitRetainedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useConnectionStatusSelector } from '@platform/transport/connectionStore';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useId, useRef, useState, type RefObject } from 'react';
import { useTranslation } from 'react-i18next';

import { WorkingMemoryCalibrateDialog } from './WorkingMemoryCalibrateDialog';
import { LineMeasurements, LineStateBadge, SharedModelsList, useCaseLabel } from './WorkingMemoryParts';

/**
 * Working-memory calibration of one model on this machine's GPU: what was measured and whether estimates use it,
 * and the runs that measure it. Every read is the server's; the section keeps no copy of a run's state, so a reload or
 * a reconnect shows the same thing.
 */
export const WorkingMemorySection = ({ modelKey }: { modelKey: string }) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const headingId = useId();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const local = useLocalAttempt(modelKey);
  const plan = useQuery(workingMemoryPlanQueryOptions(modelKey));
  const isConnected = useConnectionStatusSelector((snapshot) => snapshot.status === 'connected');
  const summary = useQuery(
    workingMemorySummaryQueryOptions(modelKey, { isConnected, isSubmitting: local?.submitting ?? false })
  );
  const status = useQuery(workingMemoryStatusQueryOptions());

  useMountEffect(() => attachWorkingMemoryRealtime(queryClient));

  const failed = [plan, summary, status].find((query) => query.isError);
  let body;

  // The plan and the summary know only installed models: this one was removed while its details were open.
  if (failed?.error instanceof ApiError && failed.error.status === 404) {
    body = (
      <Text color="fg.muted" fontSize="sm">
        {t('models.workingMemory.modelRemoved')}
      </Text>
    );
  } else if (failed) {
    body = (
      <Stack align="start" gap="2" role="alert">
        <Text color="fg.error" fontSize="sm">
          {t('models.workingMemory.loadFailed')}
          {': '}
          {getApiErrorMessage(failed.error, t('common.unknownError'))}
        </Text>
        <Button
          variant="outline"
          onClick={() => {
            for (const query of [plan, summary, status]) {
              if (query.isError) {
                void query.refetch();
              }
            }
          }}
        >
          {t('common.retry')}
        </Button>
      </Stack>
    );
  } else if (plan.data === undefined || summary.data === undefined || status.data === undefined) {
    body = (
      <Stack aria-busy="true" aria-label={t('common.loading')} gap="2">
        <Skeleton h="4" w="60%" />
        <Skeleton h="16" w="full" />
      </Stack>
    );
  } else {
    body = (
      <WorkingMemoryBody
        headingRef={headingRef}
        local={local}
        modelKey={modelKey}
        plan={plan.data}
        status={status.data}
        summary={summary.data}
      />
    );
  }

  return (
    <Stack aria-labelledby={headingId} as="section" gap="3">
      <Stack gap="0.5">
        <Text ref={headingRef} as="h3" fontSize="md" fontWeight="600" id={headingId} outline="none" tabIndex={-1}>
          {t('models.workingMemory.title')}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {t('models.workingMemory.description')}
        </Text>
      </Stack>
      {body}
    </Stack>
  );
};

const currentDeviceName = (status: WorkingMemoryStatus): string | null =>
  status.device_classes.find((entry) => entry.is_current)?.device_class?.name ?? null;

const WorkingMemoryBody = ({
  headingRef,
  local,
  modelKey,
  plan,
  status,
  summary,
}: {
  headingRef: RefObject<HTMLHeadingElement | null>;
  local: ReturnType<typeof useLocalAttempt>;
  modelKey: string;
  plan: WorkingMemoryPlan;
  status: WorkingMemoryStatus;
  summary: WorkingMemorySummary | null;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const queryClient = useQueryClient();
  const compile = useCalibrationGraphCompiler();
  const reasonId = useId();
  const primaryRef = useRef<HTMLButtonElement>(null);
  const stopRef = useRef<HTMLButtonElement>(null);
  const [isCalibrateOpen, setIsCalibrateOpen] = useState(false);
  const [isStopOpen, setIsStopOpen] = useState(false);
  const [isResetOpen, setIsResetOpen] = useState(false);
  const calibrateDialog = useExitPresence(isCalibrateOpen);
  const throughName = useModelsSelector((snapshot) =>
    plan.calibrated_through ? (snapshot.modelsByKey.get(plan.calibrated_through)?.name ?? null) : null
  );

  const steps = planCalibrationSteps(plan);
  const runs = summary?.runs ?? [];
  const { activity, outcome } = deriveCalibrationProgress({ local, runs, steps });
  const isRunning = activity.kind === 'running';
  const blocker: CalibrationBlocker | { kind: 'noCompiler' } | null =
    compile === null ? { kind: 'noCompiler' } : getCalibrationBlocker({ isRunning, plan, status, summary });
  const lines = summary?.lines ?? [];
  const coverage = describeCoverage(summary);
  const deviceName = plan.device_class?.name ?? currentDeviceName(status);
  const verified = isMeasurementMethodVerified(summary, status);
  const primaryLabel =
    outcome.kind === 'failed' || outcome.kind === 'partial'
      ? t('common.retry')
      : coverage.hasOwnMeasurements
        ? t('models.workingMemory.recalibrate')
        : t('models.workingMemory.calibrate');
  const blockerText = useBlockerText(blocker);

  // The opener of a confirmation can disappear or be disabled once it is done; the section heading stays.
  const getFinalFocus = () =>
    primaryRef.current && !primaryRef.current.disabled ? primaryRef.current : headingRef.current;
  // Starting records the attempt synchronously, so Stop has rendered by the time the closing dialog restores focus.
  const getCalibrateFinalFocus = () => stopRef.current ?? getFinalFocus();

  const confirmCalibration = () => {
    setIsCalibrateOpen(false);

    if (compile) {
      void startCalibration({ compile, modelKey, plan, queryClient });
    }
  };

  return (
    <Stack gap="3">
      <HStack align="center" gap="2" justify="space-between" wrap="wrap">
        <HStack gap="2" minW="0" wrap="wrap">
          {deviceName ? (
            <Text fontSize="sm" fontWeight="600" overflowWrap="anywhere">
              {deviceName}
            </Text>
          ) : null}
          {/* Without a current device class or a readable store nothing can be said about this GPU's measurements. */}
          {summary?.usable ? <LineStateBadge state={coverage.state} /> : null}
        </HStack>
        <HStack gap="1" wrap="wrap">
          {isRunning ? (
            <Button ref={stopRef} variant="outline" onClick={() => setIsStopOpen(true)}>
              {t('models.workingMemory.stop')}
            </Button>
          ) : null}
          {coverage.hasOwnMeasurements ? (
            <Button
              aria-describedby={isRunning && blockerText ? reasonId : undefined}
              disabled={isRunning}
              variant="ghost"
              onClick={() => setIsResetOpen(true)}
            >
              {t('common.reset')}
            </Button>
          ) : null}
          <Button
            ref={primaryRef}
            aria-describedby={blockerText ? reasonId : undefined}
            disabled={blocker !== null}
            variant="solid"
            onClick={() => setIsCalibrateOpen(true)}
          >
            {primaryLabel}
          </Button>
        </HStack>
      </HStack>

      {blockerText ? (
        <Text color="fg.muted" fontSize="xs" id={reasonId}>
          {blockerText}
        </Text>
      ) : null}

      <ActivityPanel activity={activity} />
      <OutcomePanel outcome={outcome} />

      {verified === false ? (
        <Alert.Root status="warning" variant="surface">
          <Alert.Indicator />
          <Alert.Content>
            <Alert.Title>{t('models.workingMemory.notVerified')}</Alert.Title>
          </Alert.Content>
        </Alert.Root>
      ) : null}

      {!coverage.hasCurrentLines && activity.kind === 'idle' && plan.supported && summary?.usable ? (
        <Text color="fg.muted" fontSize="sm">
          {t('models.workingMemory.neverCalibrated')}
        </Text>
      ) : null}
      {coverage.usesOthersOnly ? (
        <Text color="fg.muted" fontSize="sm">
          {t('models.workingMemory.usesOthers')}
        </Text>
      ) : null}

      {lines.length > 0 ? (
        <Stack gap="4">
          {lines.map((line) => (
            <LineMeasurements
              key={`${line.estimator}\u0000${line.line}\u0000${line.env_id}`}
              line={line}
              modelKey={modelKey}
            />
          ))}
        </Stack>
      ) : null}

      {plan.shared_with.length > 0 ? <SharedModelsList models={plan.shared_with} /> : null}

      <Stack color="fg.subtle" fontSize="xs" gap="1">
        {plan.not_covered.length > 0 ? <Text>{t('models.workingMemory.encodeNotCovered')}</Text> : null}
        <Text>{t('models.workingMemory.floorNote')}</Text>
      </Stack>

      {calibrateDialog.isMounted ? (
        <WorkingMemoryCalibrateDialog
          key={calibrateDialog.generation}
          calibratedThroughName={throughName}
          deviceName={deviceName}
          getFinalFocus={getCalibrateFinalFocus}
          isOpen={calibrateDialog.isOpen}
          plan={plan}
          onClose={() => setIsCalibrateOpen(false)}
          onConfirm={confirmCalibration}
          onExitComplete={calibrateDialog.release}
        />
      ) : null}
      <ConfirmDialog
        body={t('models.workingMemory.stopBody')}
        confirmLabel={t('models.workingMemory.stop')}
        finalFocusEl={getFinalFocus}
        isOpen={isStopOpen}
        title={t('models.workingMemory.stopTitle')}
        onClose={() => setIsStopOpen(false)}
        onConfirm={async () => {
          try {
            await stopCalibration({ modelKey, queryClient });
          } catch (error) {
            notify.error(t('models.workingMemory.stopFailed'), getApiErrorMessage(error, ''));
          }
        }}
      />
      <ConfirmDialog
        body={t('models.workingMemory.resetBody')}
        confirmLabel={t('common.reset')}
        finalFocusEl={getFinalFocus}
        isOpen={isResetOpen}
        title={t('models.workingMemory.resetTitle')}
        onClose={() => setIsResetOpen(false)}
        onConfirm={async () => {
          try {
            await resetCalibration({ modelKey, queryClient });
            notify.success(t('models.workingMemory.resetDone'));
          } catch (error) {
            notify.error(t('models.workingMemory.resetFailed'), getApiErrorMessage(error, ''));
          }
        }}
      />
    </Stack>
  );
};

const useBlockerText = (blocker: CalibrationBlocker | { kind: 'noCompiler' } | null): string | null => {
  const { t } = useTranslation();

  switch (blocker?.kind) {
    case undefined:
      return null;
    case 'unsupported':
      return blocker.reason ?? t('models.workingMemory.blockedUnsupported');
    case 'unavailable':
      return blocker.reason ?? t('models.workingMemory.blockedUnavailable');
    case 'storeUnusable':
      return t('models.workingMemory.blockedStoreUnusable', { problem: blocker.reason ?? '' });
    case 'running':
      return t('models.workingMemory.blockedRunning');
    case 'noCompiler':
      return t('models.workingMemory.blockedUnavailable');
  }
};

const ActivityPanel = ({ activity }: { activity: CalibrationActivity }) => {
  const { t } = useTranslation();
  const caseLabel = useCaseLabel();

  if (activity.kind !== 'running') {
    return null;
  }

  const { currentCase, done, queuing, total } = activity;
  const caption = queuing
    ? t('models.workingMemory.queuing', { done, total })
    : t('models.workingMemory.running', { done, total });

  return (
    <Stack gap="1">
      <Progress.Root colorPalette="accent" max={Math.max(total, 1)} value={done}>
        {/* The track is the progressbar; without a name of its own it would be announced by its percentage. */}
        <Progress.Track aria-label={t('models.workingMemory.progressLabel')} rounded="full">
          <Progress.Range rounded="full" transition="width var(--wb-motion-duration-fast) ease" />
        </Progress.Track>
      </Progress.Root>
      <Text aria-live="polite" color="fg.muted" fontSize="xs">
        {caption}
        {currentCase ? ` · ${caseLabel(currentCase)}` : null}
      </Text>
    </Stack>
  );
};

const OutcomePanel = ({ outcome }: { outcome: CalibrationOutcome }) => {
  const { t } = useTranslation();

  if (outcome.kind === 'none' || outcome.kind === 'completed') {
    return null;
  }

  if (outcome.kind === 'canceled') {
    return (
      <Text color="fg.muted" fontSize="sm">
        {t('models.workingMemory.canceled')}
      </Text>
    );
  }

  const isFailed = outcome.kind === 'failed';
  const detail =
    outcome.error ||
    (outcome.kind === 'partial'
      ? !outcome.canceled && outcome.queued < outcome.total
        ? t('models.workingMemory.partialQueued', { queued: outcome.queued, total: outcome.total })
        : t('models.workingMemory.partialCanceled')
      : null);

  return (
    <Alert.Root status={isFailed ? 'error' : 'warning'} variant="surface">
      <Alert.Indicator />
      <Alert.Content>
        <Alert.Title>{isFailed ? t('models.workingMemory.failed') : t('models.workingMemory.partial')}</Alert.Title>
        {detail ? <Alert.Description overflowWrap="anywhere">{detail}</Alert.Description> : null}
      </Alert.Content>
    </Alert.Root>
  );
};
