import type { QueryClient } from '@tanstack/react-query';

import {
  calibrationIdempotencyKey,
  formatAttemptId,
  isCalibrationOrigin,
  planCalibrationSteps,
  type CalibrationGraphCompiler,
  type LocalAttempt,
  type WorkingMemoryPlan,
  type WorkingMemoryRunCreated,
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
import { workingMemoryKeys, workingMemorySummaryQueryOptions } from './workingMemoryQueries';

/**
 * The calibration attempts this tab started, by model key: what only this tab knows (that it is still compiling or
 * posting steps, how many it meant to post, why it stopped). Every step's run carries the attempt id, so the server's
 * runs say which runs belong together, after a reload too. Kept outside the section so leaving a model's details
 * while an attempt queues, and coming back, shows the same attempt.
 */
const attemptsStore = createExternalStore<{ byModel: ReadonlyMap<string, LocalAttempt> }>({ byModel: new Map() });

interface Flight {
  promise: Promise<void>;
  stopRequested: boolean;
  /** Aborts compiling; a step being posted is not aborted, since its run exists once the response arrives. */
  stop: AbortController;
}

/** The attempt queuing per model; at most one, so a double click cannot post a second attempt. */
const flights = new Map<string, Flight>();

registerAccountOwnedResource({
  clear: () => {
    for (const flight of flights.values()) {
      flight.stopRequested = true;
      flight.stop.abort();
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

/** A lost response is resent with the same key, which returns the run the first request created. */
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
      // The server answered: resending the same request gets the same answer.
      if (error instanceof ApiError || signal.aborted || attempt >= POST_ATTEMPTS) {
        throw error;
      }
      await sleep(POST_RETRY_DELAY_MS, signal);
    }
  }
};

/**
 * Every model's summary moves: lines are shared across models, and a reset or new points change their fits. The
 * changed model's plan prices on those fits too.
 */
const invalidateAfterChange = async (queryClient: QueryClient, owner: AccountScope, modelKey?: string) => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: workingMemoryKeys.summaries(owner) }),
    queryClient.invalidateQueries({ queryKey: workingMemoryKeys.status(owner) }),
    modelKey === undefined
      ? undefined
      : queryClient.invalidateQueries({ queryKey: workingMemoryKeys.plan(owner, modelKey) }),
  ]);
};

/**
 * Compiles the plan's graphs (loading the compiler on first use) and posts every step in order, each after the last
 * one's response, so the queue runs them in step order. Stops at the first error the server answers with and keeps
 * the runs already queued: they run, and the section reports the attempt as partial.
 */
export const startCalibration = ({
  compile,
  modelKey,
  plan,
  queryClient,
}: {
  compile: CalibrationGraphCompiler;
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
  const flight: Flight = { promise: Promise.resolve(), stop: new AbortController(), stopRequested: false };

  setAttempt(modelKey, { attemptId, stopped: false, submissionError: null, submitting: true, total: steps.length });

  const queue = async () => {
    const compiling = compile(plan.cases, AbortSignal.any([owner.signal, flight.stop.signal]));
    const graphs = new Map((await compiling).map((compiled) => [compiled.caseId, compiled]));

    for (const step of steps) {
      if (flight.stopRequested || !isAccountScopeCurrent(owner)) {
        return;
      }

      const compiled = graphs.get(step.case.id);

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
      // A stop that aborted compiling is not an error.
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
        void invalidateAfterChange(queryClient, owner, modelKey);
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
    flight.stop.abort();
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
      await invalidateAfterChange(queryClient, owner, modelKey);
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
  await invalidateAfterChange(queryClient, owner, modelKey);
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
