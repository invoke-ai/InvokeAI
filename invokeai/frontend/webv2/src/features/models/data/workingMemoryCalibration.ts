import type { QueryClient } from '@tanstack/react-query';

import {
  calibrationIdempotencyKey,
  compiledModelIdentities,
  isCalibrationOrigin,
  modelCalibratingOnGpu,
  newestRunId,
  planCalibrationSteps,
  type CalibrationGraphCompiler,
  type CalibrationRefusal,
  type CompiledCalibration,
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
import { ApiError, getApiErrorCode, getApiErrorMessage, sleep } from '@platform/transport/http';
import { socketHub } from '@platform/transport/socketHub';

import { listModels } from './api';
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
  workingMemoryStatusQueryOptions,
  workingMemorySummaryQueryOptions,
} from './workingMemoryQueries';

/**
 * The calibration attempts this tab started, by model key: what only this tab knows (that it is still posting steps,
 * how many it meant to post, why it stopped). Every step's run carries the attempt id, so the server's
 * runs say which runs belong together, after a reload too. Kept outside the section so leaving a model's details
 * while an attempt queues, and coming back, shows the same attempt.
 */
const attemptsStore = createExternalStore<{ byModel: ReadonlyMap<string, LocalAttempt> }>({ byModel: new Map() });

/** How queuing an attempt ended for the tab that asked: `alreadyRunning` when the server refused its first step. */
export interface CalibrationStart {
  alreadyRunning: boolean;
}

interface Flight {
  promise: Promise<CalibrationStart>;
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
 * The refusals of a step the section explains, by the `code` of the server's structured 409 detail: another attempt
 * runs on this GPU or is between its steps, another calibration ran on it since this attempt began, or a model the
 * graph names changed or was removed since it compiled (or since an earlier step). Every other 409 has a plain detail,
 * which names no code. Each ends the attempt where it stands.
 */
const REFUSAL_CODES: Readonly<Record<string, CalibrationRefusal>> = {
  calibration_interleaved: 'interleaved',
  calibration_running: 'alreadyRunning',
  model_changed: 'modelChanged',
};

const refusalOf = (error: unknown): CalibrationRefusal | undefined =>
  error instanceof ApiError && error.status === 409 ? REFUSAL_CODES[getApiErrorCode(error) ?? ''] : undefined;

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

/**
 * The model whose calibration holds this GPU as the server sees it now, from any tab, or null when none does
 * (`modelCalibratingOnGpu`). Read right before an attempt starts, after its graphs compiled; the status query keeps
 * the read, so the section names that model, or holds a failed read's error, which the section shows. A running
 * attempt marks every model's summary stale, so an attempt of the shown model appears with its progress.
 */
export const calibrationOnGpuNow = async (queryClient: QueryClient): Promise<string | null> => {
  const owner = captureAccountScope();
  const busy = modelCalibratingOnGpu(
    await queryClient.fetchQuery({ ...workingMemoryStatusQueryOptions(owner), staleTime: 0 })
  );

  if (busy !== null) {
    await queryClient.invalidateQueries({ queryKey: workingMemoryKeys.summaries(owner) });
  }
  return busy;
};

const fetchCurrentWorkingMemorySummary = (queryClient: QueryClient, modelKey: string, owner: AccountScope) =>
  queryClient.fetchQuery({
    ...workingMemorySummaryQueryOptions(modelKey, { isConnected: true, isSubmitting: false }, owner),
    staleTime: 0,
  });

/** The model's plan as the server builds it now, for a calibration about to start. Rejects with the read's error. */
export const fetchCurrentWorkingMemoryPlan = (queryClient: QueryClient, modelKey: string): Promise<WorkingMemoryPlan> =>
  queryClient.fetchQuery({ ...workingMemoryPlanQueryOptions(modelKey), staleTime: 0 });

/**
 * The graphs of the plan's cases, as Generate would compile them over the model records the server holds now, each
 * with the identity of every model it names from those same records; loads the compiler on first use. Each step posts
 * those identities, never a later read, so the server refuses a step whose models changed since they compiled. Rejects
 * with the compiler's error, naming the first case that cannot compile, such as one whose model needs a text encoder
 * that is not installed, or with the error of reading the records. A start needs them, so compiling first is how a
 * plan the server supports turns out to be one this machine cannot run.
 */
export const compileWorkingMemoryPlan = async (
  compile: CalibrationGraphCompiler,
  plan: WorkingMemoryPlan,
  signal?: AbortSignal
): Promise<CompiledCalibration[]> => {
  const owner = captureAccountScope();
  const compileSignal = signal ? AbortSignal.any([owner.signal, signal]) : owner.signal;

  compileSignal.throwIfAborted();
  // Not this tab's models store: another tab or a script may have changed a record since it loaded, and a model's
  // base edited under the same key can plan the same SD cases while the run executes as the record the server holds.
  // Only the records, not the library's missing-file check, which stats every model's files on the server.
  const records = listModels(compileSignal);
  // The compiler awaits the records only once it has loaded; until then a failed read must not count as unhandled.
  void records.catch(() => undefined);

  const graphs = await compile(plan.cases, records, compileSignal);
  const compiledOver = await records;

  assertAccountScopeCurrent(owner);
  return graphs.map((compiled) => ({ ...compiled, models: compiledModelIdentities(compiled.graph, compiledOver) }));
};

/**
 * Posts every step of the plan with its compiled graph (`compileWorkingMemoryPlan`), in order and each after the last
 * one's response, so the queue runs them in step order. Every step names the attempt and its step count, so the server
 * fits the attempt's windows only once all of its steps have run. Stops at the first step that still fails once its
 * retries are spent: the runs already queued still run, but the server discards the attempt as aborted, and the
 * section reports it as partial. A first step the server refuses because another attempt runs on this GPU leaves
 * no record of this one: the section follows what the server lists instead.
 */
export const startCalibration = ({
  graphs,
  modelKey,
  plan,
  queryClient,
}: {
  graphs: readonly CompiledCalibration[];
  modelKey: string;
  plan: WorkingMemoryPlan;
  queryClient: QueryClient;
}): Promise<CalibrationStart> => {
  const running = flights.get(modelKey);

  if (running) {
    return running.promise;
  }

  const owner = captureAccountScope();
  const steps = planCalibrationSteps(plan);
  const attemptId = createUuid();
  const flight: Flight = { promise: Promise.resolve({ alreadyRunning: false }), stopRequested: false };
  let queued = 0;
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
      await postStep(
        modelKey,
        {
          attemptId,
          attemptSteps: steps.length,
          data: compiled.data,
          graph: compiled.graph,
          graphModels: compiled.models,
          idempotencyKey,
        },
        owner.signal
      );

      queued += 1;
      assertAccountScopeCurrent(owner);
      // Best effort: an unacknowledged receipt only expires later.
      void acknowledgeCalibrationEnqueue(idempotencyKey, owner.signal).catch(() => undefined);
      void queryClient.invalidateQueries({ queryKey: workingMemoryKeys.summary(owner, modelKey) });
    }
  };

  flight.promise = queue()
    .then(() => ({ alreadyRunning: false }))
    .catch((error: unknown) => {
      const refusal = refusalOf(error);

      // Another attempt runs on this GPU: this one queued nothing, so it leaves nothing to report.
      if (refusal === 'alreadyRunning' && queued === 0) {
        if (isAccountScopeCurrent(owner)) {
          setAttempt(modelKey, null);
        }
        return { alreadyRunning: true };
      }

      // A step that fails once a stop was asked for leaves the attempt stopped, not failed.
      if (isAccountScopeCurrent(owner) && !flight.stopRequested) {
        updateAttempt(modelKey, attemptId, (attempt) => ({
          ...attempt,
          submissionError: getApiErrorMessage(error, ''),
          submissionRefusal: refusal,
        }));
      }
      return { alreadyRunning: false };
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

  const summary = await fetchCurrentWorkingMemorySummary(queryClient, modelKey, owner);
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
