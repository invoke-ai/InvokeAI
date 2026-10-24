/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type {
  CalibrationActivity,
  CalibrationBlocker,
  CalibrationCoverage,
  CalibrationGraph,
  CalibrationGraphCompiler,
  CalibrationOutcome,
  WorkingMemoryPlan,
  WorkingMemoryStatus,
  WorkingMemorySummary,
} from '@features/models/core/workingMemory';

import { Alert, Box, Collapsible, HStack, Progress, Skeleton, Stack, Text } from '@chakra-ui/react';
import {
  changesCalibration,
  deriveCalibrationProgress,
  describeCoverage,
  getCalibrationBlocker,
  isMeasurementMethodVerified,
  planCalibrationSteps,
} from '@features/models/core/workingMemory';
import { useModelsSelector } from '@features/models/data/modelsStore';
import {
  attachWorkingMemoryRealtime,
  compileWorkingMemoryPlan,
  fetchCurrentWorkingMemoryPlan,
  refreshWorkingMemoryPlan,
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
import { ChevronRightIcon } from 'lucide-react';
import { useId, useRef, useState, type RefObject } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

import { WorkingMemoryCalibrateDialog } from './WorkingMemoryCalibrateDialog';
import { LineMeasurements, LineStateBadge, SharedModelsList, useCaseLabel } from './WorkingMemoryParts';

const INDICATOR_OPEN_STYLES = { transform: 'rotate(90deg)' } as const;
const EMPTY_LIVE_REGION = { position: 'absolute' } as const;

/**
 * Working-memory calibration of one model on this machine's GPU: what was measured and whether estimates use it,
 * and the runs that measure it. Every read is the server's; the section keeps no copy of a run's state, so a reload or
 * a reconnect shows the same thing.
 *
 * Collapsed on every visit; its header still names the GPU and the state, or the progress of a running calibration.
 */
export const WorkingMemorySection = ({ modelKey }: { modelKey: string }) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const headingId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  // The plan measures the GPU's free memory on the server, so it loads only once the panel was opened.
  const [wasOpened, setWasOpened] = useState(false);
  const local = useLocalAttempt(modelKey);
  const plan = useQuery({ ...workingMemoryPlanQueryOptions(modelKey), enabled: wasOpened });
  const isConnected = useConnectionStatusSelector((snapshot) => snapshot.status === 'connected');
  const summary = useQuery(
    workingMemorySummaryQueryOptions(modelKey, { isConnected, isSubmitting: local?.submitting ?? false })
  );
  const status = useQuery(workingMemoryStatusQueryOptions());
  const activityCaption = useActivityCaption();

  useMountEffect(() => attachWorkingMemoryRealtime(queryClient));

  // Without the plan, an attempt's step count comes from this tab's record or the attempt id.
  const progress = deriveCalibrationProgress({
    local,
    runs: summary.data?.runs ?? [],
    steps: plan.data ? planCalibrationSteps(plan.data) : [],
  });
  const coverage = describeCoverage(summary.data ?? null);
  const deviceName = plan.data?.device_class?.name ?? (status.data ? currentDeviceName(status.data) : null);
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
  } else if (!wasOpened) {
    body = null;
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
        coverage={coverage}
        deviceName={deviceName}
        focusFallbackRef={triggerRef}
        modelKey={modelKey}
        plan={plan.data}
        progress={progress}
        status={status.data}
        summary={summary.data}
      />
    );
  }

  return (
    <Collapsible.Root
      open={isOpen}
      onOpenChange={({ open }) => {
        setIsOpen(open);
        if (open) {
          setWasOpened(true);
          // Every opening reads the plan again: the GPU's free memory may have moved since the last one. The table
          // keeps the plan it has meanwhile.
          void refreshWorkingMemoryPlan(queryClient, modelKey);
        }
      }}
    >
      <Stack aria-labelledby={headingId} as="section" gap="3">
        <Stack gap="0.5">
          <HStack align="center" gap="2" justify="space-between" wrap="wrap">
            <Text as="h3" fontSize="md" fontWeight="600" id={headingId}>
              <Collapsible.Trigger ref={triggerRef} alignItems="center" display="flex" gap="1" rounded="sm">
                <Collapsible.Indicator
                  _open={INDICATOR_OPEN_STYLES}
                  transition="transform var(--wb-motion-duration-medium)"
                >
                  <ChevronRightIcon size="14" />
                </Collapsible.Indicator>
                {t('models.workingMemory.title')}
              </Collapsible.Trigger>
            </Text>
            {/* Collapsed, the header says what the open panel would; open, the panel's own row does. */}
            {!isOpen ? (
              <HStack gap="2" minW="0" wrap="wrap">
                {deviceName ? (
                  <Text color="fg.muted" fontSize="sm" overflowWrap="anywhere">
                    {deviceName}
                  </Text>
                ) : null}
                {progress.activity.kind === 'running' ? (
                  <Text color="fg.muted" fontSize="xs">
                    {activityCaption(progress.activity)}
                  </Text>
                ) : summary.data?.usable ? (
                  <LineStateBadge state={coverage.state} />
                ) : null}
              </HStack>
            ) : null}
          </HStack>
          <Text color="fg.subtle" fontSize="xs">
            {t('models.workingMemory.description')}
          </Text>
        </Stack>
        <Collapsible.Content>{body}</Collapsible.Content>
      </Stack>
    </Collapsible.Root>
  );
};

const currentDeviceName = (status: WorkingMemoryStatus): string | null =>
  status.device_classes.find((entry) => entry.is_current)?.device_class?.name ?? null;

const WorkingMemoryBody = ({
  coverage,
  deviceName,
  focusFallbackRef,
  modelKey,
  plan,
  progress: { activity, outcome },
  status,
  summary,
}: {
  coverage: CalibrationCoverage;
  deviceName: string | null;
  /** Takes focus back when a confirmation's opener is gone or disabled. */
  focusFallbackRef: RefObject<HTMLButtonElement | null>;
  modelKey: string;
  plan: WorkingMemoryPlan;
  progress: ReturnType<typeof deriveCalibrationProgress>;
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
  const [isPreparing, setIsPreparing] = useState(false);
  const [isConfirming, setIsConfirming] = useState(false);
  // How often a confirmation found the plan changed in this dialog; each change is announced anew.
  const [planChanges, setPlanChanges] = useState(0);
  // Why the plan read last could not compile; shown as the reason Calibrate opened nothing.
  const [compileError, setCompileError] = useState<string | null>(null);
  // The confirmation under way; closing the dialog aborts it, which takes it back.
  const confirmation = useRef<AbortController | null>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const [isStopOpen, setIsStopOpen] = useState(false);
  const [isResetOpen, setIsResetOpen] = useState(false);
  const calibrateDialog = useExitPresence(isCalibrateOpen);
  const throughName = useModelsSelector((snapshot) =>
    plan.calibrated_through ? (snapshot.modelsByKey.get(plan.calibrated_through)?.name ?? null) : null
  );

  const isRunning = activity.kind === 'running';
  const blocker: CalibrationBlocker | { kind: 'noCompiler' } | null =
    compile === null ? { kind: 'noCompiler' } : getCalibrationBlocker({ isRunning, plan, status, summary });
  const lines = summary?.lines ?? [];
  const verified = isMeasurementMethodVerified(summary, status);
  const primaryLabel =
    outcome.kind === 'failed' || outcome.kind === 'partial'
      ? t('common.retry')
      : coverage.hasOwnMeasurements
        ? t('models.workingMemory.recalibrate')
        : t('models.workingMemory.calibrate');
  const blockerText = useBlockerText(blocker);
  // A plan that does not compile leaves Calibrate enabled: installing what is missing is fixed outside this section,
  // and the next click checks again.
  const reasonText =
    blockerText ?? (compileError ? t('models.workingMemory.blockedCompile', { problem: compileError }) : null);

  // The opener of a confirmation can disappear or be disabled once it is done; the section's toggle stays.
  const getFinalFocus = () =>
    primaryRef.current && !primaryRef.current.disabled ? primaryRef.current : focusFallbackRef.current;
  // Starting records the attempt synchronously, so Stop has rendered by the time the closing dialog restores focus.
  const getCalibrateFinalFocus = () => stopRef.current ?? getFinalFocus();

  // Reads the plan as the server builds it now (its sizes follow the GPU's free memory and the stored measurements) and
  // compiles its cases over the model records the server holds now. Graphs compiled earlier are never reused: equal
  // cases can name a model whose record changed meanwhile. The section renders the plan read. A plan that no longer
  // holds is `unavailable`: the section shows the new reason, or the read's error with a retry. A plan the server
  // supports can still fail to compile here, say for a text encoder that is not installed.
  const readCalibration = async (
    compiler: CalibrationGraphCompiler,
    signal?: AbortSignal
  ): Promise<
    | { kind: 'unavailable' }
    | { kind: 'uncompilable'; message: string }
    | { graphs: CalibrationGraph[]; kind: 'ready'; plan: WorkingMemoryPlan }
  > => {
    const current = await fetchCurrentWorkingMemoryPlan(queryClient, modelKey).catch(() => null);

    if (current === null || getCalibrationBlocker({ isRunning, plan: current, status, summary }) !== null) {
      return { kind: 'unavailable' };
    }

    try {
      return { graphs: await compileWorkingMemoryPlan(compiler, current, signal), kind: 'ready', plan: current };
    } catch (error) {
      return { kind: 'uncompilable', message: getApiErrorMessage(error, t('common.unknownError')) };
    }
  };

  // The plan shown so far may be minutes old. Only a plan that still holds and compiles opens the dialog.
  const prepareCalibration = async () => {
    if (!compile) {
      return;
    }

    setIsPreparing(true);
    setCompileError(null);
    const readiness = await readCalibration(compile);

    if (readiness.kind === 'ready') {
      setIsPreparing(false);
      setPlanChanges(0);
      setIsCalibrateOpen(true);
      return;
    }

    // The busy Calibrate dropped focus and no dialog takes it, so return it unless it moved on meanwhile: to Calibrate,
    // enabled again, beside a compile error; else to the section's toggle, since the plan read is about to disable
    // Calibrate for its reason, or replace this panel with its error (the query tells the section a tick later).
    flushSync(() => {
      setIsPreparing(false);
      setCompileError(readiness.kind === 'uncompilable' ? readiness.message : null);
    });
    if (document.activeElement === document.body) {
      (readiness.kind === 'uncompilable' ? primaryRef.current : focusFallbackRef.current)?.focus();
    }
  };

  const closeCalibrate = () => {
    confirmation.current?.abort();
    confirmation.current = null;
    setIsConfirming(false);
    setIsCalibrateOpen(false);
  };

  // The dialog may have stood open while the GPU's free memory or the stored measurements moved: read the plan once
  // more, and start what that read returns. A plan that no longer holds or compiles closes the dialog for the
  // section's reason. One that changes what was confirmed (`changesCalibration`) starts nothing: the dialog shows it,
  // flagged, to be confirmed again. New sizes alone start: a running generation moves them, and the dialog says sizes
  // follow free memory. An unmount meanwhile does not take the confirmation back; only closing the dialog does.
  const confirmCalibration = async () => {
    if (!compile || confirmation.current) {
      return;
    }

    const controller = new AbortController();
    const confirmed = plan;
    confirmation.current = controller;
    setIsConfirming(true);
    const readiness = await readCalibration(compile, controller.signal);

    if (confirmation.current !== controller) {
      return;
    }

    confirmation.current = null;

    if (readiness.kind === 'ready' && changesCalibration(confirmed, readiness.plan)) {
      // The busy Calibrate dropped focus: enable it first, then return focus to it unless focus moved on meanwhile.
      flushSync(() => {
        setIsConfirming(false);
        setPlanChanges((count) => count + 1);
      });
      if (document.activeElement === document.body) {
        confirmRef.current?.focus();
      }
      return;
    }

    setIsConfirming(false);
    setIsCalibrateOpen(false);

    if (readiness.kind === 'uncompilable') {
      setCompileError(readiness.message);
    } else if (readiness.kind === 'ready') {
      void startCalibration({ graphs: readiness.graphs, modelKey, plan: readiness.plan, queryClient });
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
            aria-describedby={reasonText ? reasonId : undefined}
            disabled={blocker !== null}
            loading={isPreparing}
            variant="solid"
            onClick={() => void prepareCalibration()}
          >
            {primaryLabel}
          </Button>
        </HStack>
      </HStack>

      {blockerText ? (
        <Text color="fg.muted" fontSize="xs" id={reasonId}>
          {blockerText}
        </Text>
      ) : reasonText ? (
        // Announced: the click that asked for the dialog opened nothing.
        <Text color="fg.error" fontSize="xs" id={reasonId} overflowWrap="anywhere" role="alert">
          {reasonText}
        </Text>
      ) : null}

      <ActivityPanel activity={activity} />
      {/* Always present, so an outcome that appears (a stop, a failed submission) is announced; one already shown when
          the section opens is part of the page and is not. Empty, it leaves the stack's flow and adds no gap. */}
      <Box _empty={EMPTY_LIVE_REGION} role="status">
        <OutcomePanel outcome={outcome} />
      </Box>

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
          confirmRef={confirmRef}
          deviceName={deviceName}
          getFinalFocus={getCalibrateFinalFocus}
          isConfirming={isConfirming}
          isOpen={calibrateDialog.isOpen}
          plan={plan}
          planChanges={planChanges}
          onClose={closeCalibrate}
          onConfirm={() => void confirmCalibration()}
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

/** "Queuing: 2/18" while steps are posted, then "Calibrating: 5/18 runs". */
const useActivityCaption = () => {
  const { t } = useTranslation();

  return ({ done, queuing, total }: Extract<CalibrationActivity, { kind: 'running' }>): string =>
    queuing ? t('models.workingMemory.queuing', { done, total }) : t('models.workingMemory.running', { done, total });
};

const ActivityPanel = ({ activity }: { activity: CalibrationActivity }) => {
  const { t } = useTranslation();
  const caseLabel = useCaseLabel();
  const activityCaption = useActivityCaption();

  if (activity.kind !== 'running') {
    return null;
  }

  const { currentCase, done, total } = activity;
  const caption = activityCaption(activity);

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

  // Canceled before anything was measured: nothing is lost.
  if (outcome.kind === 'canceled') {
    return (
      <Text color="fg.muted" fontSize="sm">
        {t('models.workingMemory.canceled')}
      </Text>
    );
  }

  // An aborted attempt feeds no fit, the steps it finished included; Calibrate starts a new one.
  const discarded = outcome.discarded ? t('models.workingMemory.discarded') : null;
  // A step that failed ends its run, not the attempt: what the other steps measured still counts.
  const kept = outcome.kind === 'failed' && outcome.kept ? t('models.workingMemory.failedKept') : null;

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
        {discarded ? <Alert.Description>{discarded}</Alert.Description> : null}
        {kept ? <Alert.Description>{kept}</Alert.Description> : null}
      </Alert.Content>
    </Alert.Root>
  );
};
