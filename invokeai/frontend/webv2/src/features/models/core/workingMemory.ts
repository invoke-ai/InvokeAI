/** Calibration step order, retry keys and attempt grouping; the server owns every measurement and state. */

import type { components } from '@api/schema';

import type { ModelConfig } from './types';

type Schemas = components['schemas'];

export type WorkingMemoryPlan = Schemas['WorkingMemoryCalibrationPlan'];
export type WorkingMemoryCase = Schemas['WorkingMemoryCalibrationCase'];
export type WorkingMemorySummary = Schemas['WorkingMemoryCalibrationModelSummary'];
export type WorkingMemoryLine = Schemas['WorkingMemoryCalibrationModelLine'];
export type WorkingMemoryLineSize = Schemas['WorkingMemoryCalibrationLineSize'];
export type WorkingMemoryRun = Schemas['WorkingMemoryCalibrationModelRun'];
export type WorkingMemoryStatus = Schemas['WorkingMemoryCalibrationStatus'];
export type WorkingMemoryRunCreated = Schemas['WorkingMemoryCalibrationRunCreated'];
export type WorkingMemoryLineState = WorkingMemoryLine['state'];
export type WorkingMemoryLineContributor = Schemas['WorkingMemoryCalibrationLineContributor'];
/** As the compiled-over record has it; the server compares it with the model's record (`graph_models`). */
export type CalibrationCompiledModel = Schemas['CalibrationCompiledModel'];

/** The server reserves this origin prefix for calibration queue items; the run id follows it. */
export const CALIBRATION_ORIGIN_PREFIX = 'invokeai:wm-calibration:';

export const calibrationRunOrigin = (runId: string): string => `${CALIBRATION_ORIGIN_PREFIX}${runId}`;

export const isCalibrationOrigin = (origin: unknown): boolean =>
  typeof origin === 'string' && origin.startsWith(CALIBRATION_ORIGIN_PREFIX);

/** Models whose plan the section asks for: calibration covers main models and standalone VAEs. */
export const supportsWorkingMemoryCalibration = (model: { type: string }): boolean =>
  model.type === 'main' || model.type === 'vae';

export interface CalibrationGraph {
  caseId: string;
  graph: unknown;
  data: unknown;
}

export interface CompiledCalibration extends CalibrationGraph {
  models: Readonly<Record<string, CalibrationCompiledModel>>;
}

type JsonObject = Record<string, unknown>;

const isJsonObject = (value: unknown): value is JsonObject =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** `ModelIdentifierField` shape; the server's enum checks are skipped, since no list here stays current. */
const isModelIdentifier = (value: JsonObject): value is JsonObject & { key: string } =>
  ['key', 'hash', 'name', 'base', 'type'].every((field) => typeof value[field] === 'string') &&
  (value.submodel_type === undefined || value.submodel_type === null || typeof value.submodel_type === 'string');

/** Every model key the nodes name, nested ones included, as the server's `graph_model_identifiers` finds them. */
const modelKeysOf = (graph: unknown): Set<string> => {
  const keys = new Set<string>();
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      value.forEach(visit);
    } else if (isJsonObject(value)) {
      if (isModelIdentifier(value)) {
        keys.add(value.key);
      } else {
        Object.values(value).forEach(visit);
      }
    }
  };
  const nodes = isJsonObject(graph) && isJsonObject(graph.nodes) ? Object.values(graph.nodes) : [];

  for (const node of nodes) {
    if (isJsonObject(node)) {
      Object.values(node).forEach(visit);
    }
  }
  return keys;
};

/** Throws for a model `records` lack, which a compiler over them cannot have named. */
export const compiledModelIdentities = (
  graph: unknown,
  records: readonly ModelConfig[]
): Record<string, CalibrationCompiledModel> => {
  const byKey = new Map(records.map((record) => [record.key, record]));
  const identities: Record<string, CalibrationCompiledModel> = {};

  for (const key of modelKeysOf(graph)) {
    const record = byKey.get(key);

    if (!record) {
      throw new Error(`The graph names model ${key}, which the records it was compiled over lack.`);
    }
    identities[key] = {
      base: record.base,
      format: record.format,
      hash: record.hash,
      type: record.type,
      variant: record.variant ?? null,
    };
  }
  return identities;
};

/** `models` is still being read when called. Generation owns compilation and depends on Models, so App supplies it. */
export type CalibrationGraphCompiler = (
  cases: readonly WorkingMemoryCase[],
  models: Promise<readonly ModelConfig[]>,
  signal: AbortSignal
) => Promise<CalibrationGraph[]>;

export type CalibrationPhase = 'warm_up' | 'measured';

/** One queue item of an attempt: a case at one repetition. Each step is its own POST and its own server run. */
export interface CalibrationStep {
  /** Position in the attempt, which is also the order the server creates the step's run in. */
  index: number;
  case: WorkingMemoryCase;
  phase: CalibrationPhase;
  repetition: number;
}

/** Grouped by decode dtype, so each precision switch (it recasts the VAE; the server excludes it) hits a warm-up. */
export const planCalibrationSteps = (plan: Pick<WorkingMemoryPlan, 'cases' | 'repetitions'>): CalibrationStep[] => {
  const groups = new Map<string, WorkingMemoryCase[]>();

  for (const item of plan.cases) {
    const group = groups.get(item.line);

    if (group) {
      group.push(item);
    } else {
      groups.set(item.line, [item]);
    }
  }

  const steps: CalibrationStep[] = [];
  const push = (item: WorkingMemoryCase, phase: CalibrationPhase, count: number) => {
    for (let repetition = 1; repetition <= count; repetition += 1) {
      steps.push({ case: item, index: steps.length, phase, repetition });
    }
  };

  for (const cases of groups.values()) {
    for (const item of cases) {
      push(item, 'warm_up', plan.repetitions.warm_up);
      push(item, 'measured', plan.repetitions.measured);
    }
  }

  return steps;
};

/** Sizes follow free GPU memory, which a running generation moves, so only new lines or run counts count. */
export const changesCalibration = (
  confirmed: Pick<WorkingMemoryPlan, 'cases' | 'repetitions'>,
  current: Pick<WorkingMemoryPlan, 'cases' | 'repetitions'>
): boolean => {
  const linesOf = (plan: Pick<WorkingMemoryPlan, 'cases'>) =>
    [...new Set(plan.cases.map((item) => `${item.estimator}\u0000${item.line}`))].sort().join('\u0001');

  return (
    linesOf(confirmed) !== linesOf(current) ||
    planCalibrationSteps(confirmed).length !== planCalibrationSteps(current).length
  );
};

/** Carries the attempt id: the server keys retry receipts on the key alone, so reuse by a new attempt is refused. */
export const calibrationIdempotencyKey = (attemptId: string, step: CalibrationStep): string =>
  `webv2:wm-calibration:${attemptId}:${String(step.index)}:${step.case.id}:${step.phase}-${String(step.repetition)}`;

const byCreation = (left: WorkingMemoryRun, right: WorkingMemoryRun): number =>
  Date.parse(left.created_at) - Date.parse(right.created_at);

export const newestRunId = (runs: readonly WorkingMemoryRun[]): string | null =>
  [...runs].sort(byCreation).at(-1)?.run_id ?? null;

/** Defaults to the newest run's attempt; a run without an attempt id is an attempt of its own. */
export const selectAttemptRuns = (runs: readonly WorkingMemoryRun[], attemptId?: string): WorkingMemoryRun[] => {
  const ordered = [...runs].sort(byCreation);
  const newest = ordered.at(-1);
  const target = attemptId ?? newest?.attempt_id ?? null;

  if (target === null) {
    return newest ? [newest] : [];
  }

  return ordered.filter((run) => run.attempt_id === target);
};

/** `interleaved`: another calibration ran on this GPU between this attempt's steps, so they no longer combine. */
export type CalibrationRefusal = 'alreadyRunning' | 'interleaved' | 'modelChanged';

export interface LocalAttempt {
  attemptId: string;
  total: number;
  submitting: boolean;
  submissionError: string | null;
  submissionRefusal?: CalibrationRefusal;
  stopped: boolean;
  /** While the attempt has no runs, it is the latest only until a run newer than this appears. */
  newestRunAtStart: string | null;
}

export type CalibrationActivity =
  | { kind: 'idle' }
  | {
      kind: 'running';
      done: number;
      total: number;
      /** The case the queue is running or about to run, when it can be told. */
      currentCase: WorkingMemoryCase | null;
      queuing: boolean;
    };

/** `discarded`: the server aborted a measuring attempt, so no window feeds a fit; a failed run does not abort it. */
export type CalibrationOutcome =
  | { kind: 'none' }
  | { kind: 'completed' }
  /** Every run was canceled before it measured anything, so nothing was discarded. */
  | { kind: 'canceled' }
  /** `canceled`: runs were canceled, so the attempt did not stop because queuing ended early. */
  | {
      kind: 'partial';
      canceled: boolean;
      discarded: boolean;
      error: string | null;
      refusal?: CalibrationRefusal;
      queued: number;
      total: number;
    }
  | { kind: 'failed'; discarded: boolean; error: string | null; kept: boolean; refusal?: CalibrationRefusal };

/** A run counts as running while any item is pending, so a pending item blocks a second start across tabs. */
export const deriveCalibrationProgress = ({
  local,
  runs,
  steps,
}: {
  local: LocalAttempt | null;
  runs: readonly WorkingMemoryRun[];
  steps: readonly CalibrationStep[];
}): { activity: CalibrationActivity; outcome: CalibrationOutcome } => {
  // Another tab may have started a newer attempt; one that queued nothing stays current until a newer run exists.
  const newest = [...runs].sort(byCreation).at(-1);
  const current =
    local !== null &&
    (local.submitting ||
      (newest !== undefined && newest.attempt_id === local.attemptId) ||
      (!runs.some((run) => run.attempt_id === local.attemptId) &&
        (newest?.run_id ?? null) === local.newestRunAtStart &&
        !runs.some((run) => run.status === 'running')))
      ? local
      : null;
  const attempt = selectAttemptRuns(runs, current?.attemptId);
  // The step count from this tab, else the server's, so any tab can tell an attempt cut short from a complete one.
  const planned = current?.total ?? attempt.find((run) => run.attempt_steps !== null)?.attempt_steps ?? null;
  const total = Math.max(planned ?? steps.length, attempt.length);
  const plannedTotal = planned ?? attempt.length;
  const runningIndex = attempt.findIndex((run) => run.status === 'running');
  const queuing = current?.submitting ?? false;

  if (queuing || runningIndex !== -1) {
    // Steps settle in order: the queue runs an attempt's items first in, first out.
    const done = attempt.filter((run) => run.status !== 'running').length;
    const currentIndex = runningIndex === -1 ? attempt.length : runningIndex;

    return {
      activity: { currentCase: steps[currentIndex]?.case ?? null, done, kind: 'running', queuing, total },
      outcome: { kind: 'none' },
    };
  }

  const idle: CalibrationActivity = { kind: 'idle' };
  const canceled =
    (current?.stopped ?? false) || attempt.some((run) => run.status === 'canceled' || run.status === 'partial');
  // Once settled, only the server says whether windows are discarded; `point_count` counts invalid windows too.
  const measured = attempt.some((run) => run.status === 'completed' || run.status === 'partial');
  const aborted = attempt.some((run) => run.held_back === 'aborted_attempt');
  const discarded = measured && aborted;
  const partial = (error: string | null, refusal?: CalibrationRefusal): CalibrationOutcome => ({
    canceled,
    discarded,
    error,
    kind: 'partial',
    queued: attempt.length,
    refusal,
    total: plannedTotal,
  });

  // A submission error stands until the server lists every planned step (a lost response may have been recorded).
  if (current?.submissionError && attempt.length < plannedTotal) {
    return {
      activity: idle,
      outcome:
        attempt.length === 0
          ? {
              discarded,
              error: current.submissionError,
              kept: false,
              kind: 'failed',
              refusal: current.submissionRefusal,
            }
          : partial(current.submissionError, current.submissionRefusal),
    };
  }

  if (attempt.length === 0) {
    return { activity: idle, outcome: current?.stopped ? { kind: 'canceled' } : { kind: 'none' } };
  }

  // A graph model changed after planning: its items fail on admission, and the section says so in its own words.
  const failed =
    attempt.find((run) => run.status === 'failed' && run.error_code === 'model_changed') ??
    attempt.find((run) => run.status === 'failed');

  if (failed) {
    const modelChanged = failed.error_code === 'model_changed';

    return {
      activity: idle,
      outcome: {
        discarded,
        error: failed.error,
        // What earlier steps measured is of a model that has changed since: no claim that it still counts.
        kept: measured && !aborted && !modelChanged,
        kind: 'failed',
        refusal: modelChanged ? 'modelChanged' : undefined,
      },
    };
  }

  if (attempt.every((run) => run.status === 'canceled')) {
    return { activity: idle, outcome: { kind: 'canceled' } };
  }

  if (attempt.every((run) => run.status === 'completed') && attempt.length >= plannedTotal) {
    return { activity: idle, outcome: { kind: 'completed' } };
  }

  return { activity: idle, outcome: partial(null) };
};

/** `interrupted`: a line only aborted attempts measured, which the server calls `insufficient`. */
export type LineDisplayState = WorkingMemoryLineState | 'interrupted';

export const lineDisplayState = (line: Pick<WorkingMemoryLine, 'sizes' | 'state'>): LineDisplayState =>
  line.state === 'insufficient' &&
  line.sizes.length > 0 &&
  line.sizes.every((size) => size.excluded_reason === 'aborted_attempt')
    ? 'interrupted'
    : line.state;

const LINE_STATE_SEVERITY: readonly LineDisplayState[] = ['unusable', 'stale', 'insufficient', 'interrupted', 'in_use'];

export const summarizeLineStates = (states: readonly LineDisplayState[]): LineDisplayState | null =>
  LINE_STATE_SEVERITY.find((state) => states.includes(state)) ?? null;

export interface CalibrationCoverage {
  /** The headline state: of the current environment's lines, or of all lines when it has none. */
  state: LineDisplayState | null;
  hasCurrentLines: boolean;
  /** This model has windows on some line or has runs, so a reset removes something. */
  hasOwnMeasurements: boolean;
  usesOthersOnly: boolean;
}

export const describeCoverage = (
  summary: {
    current_env_id: string | null;
    lines: readonly Pick<
      WorkingMemoryLine,
      'env_id' | 'estimator' | 'line' | 'own_windows' | 'sizes' | 'state' | 'version'
    >[];
    runs: readonly unknown[];
  } | null
): CalibrationCoverage => {
  const lines = summary?.lines ?? [];
  const current = lines.filter((line) => line.env_id === summary?.current_env_id);
  const hasOwnWindows = lines.some((line) => line.own_windows > 0);
  const applies = (line: (typeof lines)[number]) => line.state === 'in_use' || line.state === 'stale';
  // An estimator update leaves the old line unusable; once the new version's line applies, the old one is superseded.
  const headline = current.filter(
    (line) =>
      line.state !== 'unusable' ||
      !current.some(
        (other) =>
          other.estimator === line.estimator &&
          other.line === line.line &&
          other.version !== line.version &&
          applies(other)
      )
  );

  return {
    hasCurrentLines: current.length > 0,
    hasOwnMeasurements: hasOwnWindows || (summary?.runs.length ?? 0) > 0,
    state: summarizeLineStates((current.length > 0 ? headline : lines).map(lineDisplayState)),
    usesOthersOnly: !hasOwnWindows && current.some(applies),
  };
};

/** A line's contributors with this model first, the rest in the server's order (by name). */
export const orderContributors = (
  contributors: readonly WorkingMemoryLineContributor[],
  modelKey: string
): WorkingMemoryLineContributor[] => [
  ...contributors.filter((entry) => entry.model_key === modelKey),
  ...contributors.filter((entry) => entry.model_key !== modelKey),
];

export type CalibrationBlocker =
  | { kind: 'unsupported'; reason: string | null }
  | { kind: 'unavailable'; reason: string | null }
  | { kind: 'storeUnusable'; reason: string | null }
  | { kind: 'running' };

/** Ordered by what a user can act on. */
export const getCalibrationBlocker = ({
  isRunning,
  plan,
  status,
  summary,
}: {
  isRunning: boolean;
  plan: Pick<WorkingMemoryPlan, 'reason' | 'supported'>;
  status: Pick<WorkingMemoryStatus, 'available' | 'unavailable_reason'>;
  /** Null when the server has no current device class to summarise. */
  summary: Pick<WorkingMemorySummary, 'problem' | 'usable'> | null;
}): CalibrationBlocker | null => {
  if (!plan.supported) {
    return { kind: 'unsupported', reason: plan.reason };
  }

  if (summary && !summary.usable) {
    return { kind: 'storeUnusable', reason: summary.problem };
  }

  if (!status.available || summary === null) {
    return { kind: 'unavailable', reason: status.unavailable_reason };
  }

  return isRunning ? { kind: 'running' } : null;
};

/** One attempt per device class; an attempt between steps reads as free here, and the server's refusal covers it. */
export const modelCalibratingOnGpu = (status: Pick<WorkingMemoryStatus, 'device_classes'>): string | null =>
  status.device_classes
    .find((entry) => entry.is_current)
    ?.runs.find((run) => run.status === 'running' && run.attempt_id !== null)?.model_key ?? null;

/** The model summary's answer, else the current device class's; null while neither can tell. */
export const isMeasurementMethodVerified = (
  summary: Pick<WorkingMemorySummary, 'measurement_verified'> | null,
  status: WorkingMemoryStatus
): boolean | null =>
  summary?.measurement_verified ??
  status.device_classes.find((entry) => entry.is_current)?.measurement_verified ??
  null;

/** `decode_fp16` → operation `decode`, dtype `fp16`. A line without a dtype suffix keeps its whole name. */
export const splitLine = (line: string): { dtype: string | null; operation: string } => {
  const separator = line.lastIndexOf('_');

  return separator <= 0
    ? { dtype: null, operation: line }
    : { dtype: line.slice(separator + 1), operation: line.slice(0, separator) };
};

export type TranslatedFact = 'allocator' | 'driver' | 'invokeai' | 'osBuild' | 'sdpaGuard' | 'torchBuild';

const FACT_NAMES: Readonly<Record<string, string>> = {
  cuda: 'CUDA',
  cudnn: 'cuDNN',
  hip: 'ROCm',
  miopen: 'MIOpen',
  os: 'OS',
  torch: 'PyTorch',
  xpu: 'XPU runtime',
};

const TRANSLATED_FACTS: Readonly<Record<string, TranslatedFact>> = {
  allocator_backend: 'allocator',
  driver: 'driver',
  invokeai: 'invokeai',
  os_build: 'osBuild',
  rocm_sdpa_guard: 'sdpaGuard',
  torch_build: 'torchBuild',
};

/** Product name, translated label, or the key itself for env vars and settings; unknown keys show as sent. */
export const describeFact = (key: string): { name: string } | { translated: TranslatedFact } => {
  const translated = TRANSLATED_FACTS[key];

  if (translated) {
    return { translated };
  }

  for (const prefix of ['env.', 'setting.']) {
    if (key.startsWith(prefix) && key.length > prefix.length) {
      return { name: key.slice(prefix.length) };
    }
  }

  return { name: FACT_NAMES[key] ?? key };
};
