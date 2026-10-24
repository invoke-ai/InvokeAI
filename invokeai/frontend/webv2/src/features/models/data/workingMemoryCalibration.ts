import type { QueryClient } from '@tanstack/react-query';

import {
  calibrationIdempotencyKey,
  formatAttemptId,
  isCalibrationOrigin,
  newestRunId,
  planCalibrationSteps,
  type CalibrationGraph,
  type CalibrationGraphCompiler,
  type LocalAttempt,
  type WorkingMemoryPlan,
  type WorkingMemoryRunCreated,
  type WorkingMemorySummary,
} from '@features/models/core/workingMemory';
import { createUuid } from '@platform/browser/randomUuid';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
  type AccountScope,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { ApiError, getApiErrorMessage, sleep } from '@platform/transport/http';
import { socketHub } from '@platform/transport/socketHub';

import {
  acknowledgeCalibrationEnqueue,
  calibrateWorkingMemory,
  cancelCalibrationRun,
  resetWorkingMemory,
  type CalibrateRequest,
} from './workingMemoryApi';
import {
  workingMemoryKeys,
  workingMemoryPlanQueryOptions,
  workingMemorySummaryQueryOptions,
} from './workingMemoryQueries';

/**
 * The calibration attempts this tab started, by model key: what only this tab knows (that it is still posting steps,
 * how many it meant to post, why it stopped). Every step's run carries the attempt id, so the server's
 * runs say which runs belong together, after a reload too. Kept outside the section so leaving a model's details
 * while an attempt queues, and coming back, shows the same attempt.
 */
const attemptsStore = createExternalStore<{ byModel: ReadonlyMap<string, LocalAttempt> }>({ byModel: new Map() });

interface Flight {
  promise: Promise<void>;
  /** Ends queuing before the next step; a step being posted is not aborted, since its run exists once it answers. */
  stopRequested: boolean;
}

/** The attempt queuing per model; at most one, so a double click cannot post a second attempt. */
const flights = new Map<string, Flight>();

registerAccountOwnedResource({
  clear: () => {
    for (const flight of flights.values()) {
      flight.stopRequested = true;
    }
    flights.clear();
    attemptsStore.setSnapshot({ byModel: new Map() });
  },
  name: 'working-memory-calibration',
});

const setAttempt = (modelKey: string, attempt: LocalAttempt | null): void => {
  const byModel = new Map(attemptsStore.getSnapshot().byModel);

  if (attempt) {
    byModel.set(modelKey, attempt);
  } else {
    byModel.delete(modelKey);
  }

  attemptsStore.setSnapshot({ byModel });
};

const updateAttempt = (modelKey: string, attemptId: string, change: (attempt: LocalAttempt) => LocalAttempt) => {
  const attempt = attemptsStore.getSnapshot().byModel.get(modelKey);

  // A reset or a newer attempt replaced it meanwhile.
  if (attempt?.attemptId === attemptId) {
    setAttempt(modelKey, change(attempt));
  }
};

export const useLocalAttempt = (modelKey: string): LocalAttempt | null =>
  attemptsStore.useSelector((snapshot) => snapshot.byModel.get(modelKey) ?? null, Object.is);

/**
 * A lost response is resent with the same key, which returns the run the first request created. So is a 503: the
 * store failed for a passing reason (an unwritable root, a lock timeout), possibly after the items were queued, and a
 * repeat with the key records them without queuing anything new.
 */
const POST_ATTEMPTS = 3;
const POST_RETRY_DELAY_MS = 1_000;

const postStep = async (
  modelKey: string,
  request: CalibrateRequest,
  signal: AbortSignal
): Promise<WorkingMemoryRunCreated> => {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await calibrateWorkingMemory(modelKey, request, signal);
    } catch (error) {
      // Any other answer of the server holds for a repeat of the same request.
      if ((error instanceof ApiError && error.status !== 503) || signal.aborted || attempt >= POST_ATTEMPTS) {
        throw error;
      }
      await sleep(POST_RETRY_DELAY_MS, signal);
    }
  }
};

/**
 * Every model's summary moves: lines are shared across models, and a reset or new points change their fits. So does
 * every model's plan, which prices on those fits and sizes for the memory the stored measurements found. Plans are
 * only marked stale, not read now: a read measures the GPU's free memory, which a calibration that is running or just
 * ended does not leave as a generation finds it, and a running attempt's progress names its cases from the shown plan.
 * The next open of a section and the next start read them (`refreshWorkingMemoryPlan`, `fetchCurrentWorkingMemoryPlan`).
 */
const invalidateAfterChange = async (queryClient: QueryClient, owner: AccountScope) => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: workingMemoryKeys.summaries(owner) }),
    queryClient.invalidateQueries({ queryKey: workingMemoryKeys.status(owner) }),
    queryClient.invalidateQueries({ queryKey: workingMemoryKeys.plans(owner), refetchType: 'none' }),
  ]);
};

/** Reads the model's plan again if a section shows it, else on its next read: the GPU's free memory may have moved. */
export const refreshWorkingMemoryPlan = (queryClient: QueryClient, modelKey: string): Promise<void> =>
  queryClient.invalidateQueries({ queryKey: workingMemoryPlanQueryOptions(modelKey).queryKey });

/** The model's plan as the server builds it now, for a calibration about to start. Rejects with the read's error. */
export const fetchCurrentWorkingMemoryPlan = (queryClient: QueryClient, modelKey: string): Promise<WorkingMemoryPlan> =>
  queryClient.fetchQuery({ ...workingMemoryPlanQueryOptions(modelKey), staleTime: 0 });

/**
 * The graphs of the plan's cases, as Generate would compile them over the models this tab holds now; loads the
 * compiler on first use. Rejects with the compiler's error, naming the first case that cannot compile, such as one
 * whose model needs a text encoder that is not installed. A start needs them, so compiling first is how a plan the
 * server supports turns out to be one this machine cannot run.
 */
export const compileWorkingMemoryPlan = async (
  compile: CalibrationGraphCompiler,
  plan: WorkingMemoryPlan,
  signal?: AbortSignal
): Promise<CalibrationGraph[]> => {
  const owner = captureAccountScope();
  const graphs = await compile(plan.cases, signal ? AbortSignal.any([owner.signal, signal]) : owner.signal);

  assertAccountScopeCurrent(owner);
  return graphs;
};

/**
 * Posts every step of the plan with its compiled graph (`compileWorkingMemoryPlan`), in order and each after the last
 * one's response, so the queue runs them in step order. Stops at the first step that still fails once its retries
 * are spent and keeps the runs already queued: they run, and the section reports the attempt as partial.
 */
export const startCalibration = ({
  graphs,
  modelKey,
  plan,
  queryClient,
}: {
  graphs: readonly CalibrationGraph[];
  modelKey: string;
  plan: WorkingMemoryPlan;
  queryClient: QueryClient;
}): Promise<void> => {
  const running = flights.get(modelKey);

  if (running) {
    return running.promise;
  }

  const owner = captureAccountScope();
  const steps = planCalibrationSteps(plan);
  const attemptId = formatAttemptId(createUuid(), steps.length);
  const flight: Flight = { promise: Promise.resolve(), stopRequested: false };
  const byCase = new Map(graphs.map((compiled) => [compiled.caseId, compiled]));

  const known = queryClient.getQueryData<WorkingMemorySummary | null>(workingMemoryKeys.summary(owner, modelKey));

  setAttempt(modelKey, {
    attemptId,
    newestRunAtStart: newestRunId(known?.runs ?? []),
    stopped: false,
    submissionError: null,
    submitting: true,
    total: steps.length,
  });

  const queue = async () => {
    for (const step of steps) {
      if (flight.stopRequested || !isAccountScopeCurrent(owner)) {
        return;
      }

      const compiled = byCase.get(step.case.id);

      if (!compiled) {
        throw new Error(`No graph was built for case ${step.case.id}.`);
      }

      const idempotencyKey = calibrationIdempotencyKey(attemptId, step);
      await postStep(modelKey, { attemptId, data: compiled.data, graph: compiled.graph, idempotencyKey }, owner.signal);

      assertAccountScopeCurrent(owner);
      // Best effort: an unacknowledged receipt only expires later.
      void acknowledgeCalibrationEnqueue(idempotencyKey, owner.signal).catch(() => undefined);
      void queryClient.invalidateQueries({ queryKey: workingMemoryKeys.summary(owner, modelKey) });
    }
  };

  flight.promise = queue()
    .catch((error: unknown) => {
      // A step that fails once a stop was asked for leaves the attempt stopped, not failed.
      if (isAccountScopeCurrent(owner) && !flight.stopRequested) {
        updateAttempt(modelKey, attemptId, (attempt) => ({
          ...attempt,
          submissionError: getApiErrorMessage(error, ''),
        }));
      }
    })
    .finally(() => {
      if (flights.get(modelKey) === flight) {
        flights.delete(modelKey);
      }

      if (isAccountScopeCurrent(owner)) {
        updateAttempt(modelKey, attemptId, (attempt) => ({ ...attempt, submitting: false }));
        void invalidateAfterChange(queryClient, owner);
      }
    });
  flights.set(modelKey, flight);

  return flight.promise;
};

/**
 * Stops queuing, then cancels every run of the model the queue still holds, newest first so no later item starts
 * while an earlier one is cancelled. Waits for the step in flight: its run exists once the response arrives.
 */
export const stopCalibration = async ({
  modelKey,
  queryClient,
}: {
  modelKey: string;
  queryClient: QueryClient;
}): Promise<void> => {
  const owner = captureAccountScope();
  const flight = flights.get(modelKey);

  const local = attemptsStore.getSnapshot().byModel.get(modelKey);

  if (local) {
    updateAttempt(modelKey, local.attemptId, (attempt) => ({ ...attempt, stopped: true }));
  }

  if (flight) {
    flight.stopRequested = true;
    await flight.promise;
  }

  const summary = await queryClient.fetchQuery({
    ...workingMemorySummaryQueryOptions(modelKey, { isConnected: true, isSubmitting: false }, owner),
    staleTime: 0,
  });
  const running = (summary?.runs ?? [])
    .filter((run) => run.status === 'running')
    .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at));

  try {
    for (const run of running) {
      await cancelCalibrationRun(run.run_id, owner.signal);
    }
  } finally {
    // Awaited, so a confirmation closing after this sees whether Calibrate is available again.
    if (isAccountScopeCurrent(owner)) {
      await invalidateAfterChange(queryClient, owner);
    }
  }
};

/** Removes the model's runs and measurements; the shipped estimates apply again from the next generation on. */
export const resetCalibration = async ({
  modelKey,
  queryClient,
}: {
  modelKey: string;
  queryClient: QueryClient;
}): Promise<void> => {
  const owner = captureAccountScope();

  await resetWorkingMemory(modelKey, owner.signal);
  assertAccountScopeCurrent(owner);
  setAttempt(modelKey, null);
  await invalidateAfterChange(queryClient, owner);
};

/**
 * Keeps an open section current: a calibration item changing status refreshes the summaries, and a reconnect
 * refreshes everything, since events during the gap were lost. Polling covers the time without a socket.
 */
export const attachWorkingMemoryRealtime = (queryClient: QueryClient): (() => void) => {
  const owner = captureAccountScope();

  const detachStatus = socketHub.on('queue_item_status_changed', (payload: never) => {
    const origin = (payload as { origin?: unknown } | null)?.origin;

    if (isCalibrationOrigin(origin) && isAccountScopeCurrent(owner)) {
      void invalidateAfterChange(queryClient, owner);
    }
  });

  // The hub reports the current status on subscribe; only a later transition to connected is a reconnect.
  let wasConnected: boolean | null = null;
  const detachConnection = socketHub.onConnectionChange((status) => {
    const isConnected = status === 'connected';

    if (isConnected && wasConnected === false && isAccountScopeCurrent(owner)) {
      void queryClient.invalidateQueries({ queryKey: workingMemoryKeys.forAccount(owner) });
    }
    wasConnected = isConnected;
  });

  return () => {
    detachStatus();
    detachConnection();
  };
};
