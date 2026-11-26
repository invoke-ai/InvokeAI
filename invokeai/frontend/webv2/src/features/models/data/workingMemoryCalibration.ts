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

import { listModelsWithAvailability } from './modelsStore';
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

/** Tab-local attempt state by model key; kept outside the section so leaving and returning shows the same attempt. */
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

/** By the `code` of the server's structured 409 detail; other 409s carry no code. Each ends the attempt. */
const REFUSAL_CODES: Readonly<Record<string, CalibrationRefusal>> = {
  calibration_interleaved: 'interleaved',
  calibration_running: 'alreadyRunning',
  model_changed: 'modelChanged',
};

const refusalOf = (error: unknown): CalibrationRefusal | undefined =>
  error instanceof ApiError && error.status === 409 ? REFUSAL_CODES[getApiErrorCode(error) ?? ''] : undefined;

/** Lost responses and 503s are resent with the same key, which returns or records the first request's run. */
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

/** Every summary and plan moves (lines are shared). Plans are only marked stale: a read now measures skewed memory. */
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

/** Read right before an attempt starts, after its graphs compiled. */
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

export const fetchCurrentWorkingMemoryPlan = (queryClient: QueryClient, modelKey: string): Promise<WorkingMemoryPlan> =>
  queryClient.fetchQuery({ ...workingMemoryPlanQueryOptions(modelKey), staleTime: 0 });

/** Steps post these compiled identities, never a later read, so the server refuses a step whose models changed. */
export const compileWorkingMemoryPlan = async (
  compile: CalibrationGraphCompiler,
  plan: WorkingMemoryPlan,
  signal?: AbortSignal
): Promise<CompiledCalibration[]> => {
  const owner = captureAccountScope();
  const compileSignal = signal ? AbortSignal.any([owner.signal, signal]) : owner.signal;

  compileSignal.throwIfAborted();
  // Not this tab's models store: another tab may have changed a record. Records with current file availability.
  const records = listModelsWithAvailability(compileSignal);
  // The compiler awaits the records only once it has loaded; until then a failed read must not count as unhandled.
  void records.catch(() => undefined);

  const graphs = await compile(plan.cases, records, compileSignal);
  const compiledOver = await records;

  assertAccountScopeCurrent(owner);
  return graphs.map((compiled) => ({ ...compiled, models: compiledModelIdentities(compiled.graph, compiledOver) }));
};

/** Posts steps sequentially so the queue runs them in order; the server fits windows only once every step ran. */
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

/** Cancels newest first so no later item starts while an earlier one is cancelled; waits for the step in flight. */
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

/** A reconnect refreshes everything, since events during the gap were lost; polling covers time without a socket. */
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
