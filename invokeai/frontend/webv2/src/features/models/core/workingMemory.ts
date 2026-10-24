/**
 * Working-memory calibration as the Model Manager sees it: the order a plan's cases run in, the retry keys that make a
 * lost response safe to resend, which of a model's runs make up the latest attempt, and what the section shows for
 * a plan, a summary and the device status. The server owns every measurement and every state; this module only
 * orders, groups and reads them.
 */

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

/** The server reserves this origin prefix for calibration queue items; the run id follows it. */
export const CALIBRATION_ORIGIN_PREFIX = 'invokeai:wm-calibration:';

export const calibrationRunOrigin = (runId: string): string => `${CALIBRATION_ORIGIN_PREFIX}${runId}`;

export const isCalibrationOrigin = (origin: unknown): boolean =>
  typeof origin === 'string' && origin.startsWith(CALIBRATION_ORIGIN_PREFIX);

/** Models whose plan the section asks for: calibration covers main models and standalone VAEs. */
export const supportsWorkingMemoryCalibration = (model: { type: string }): boolean =>
  model.type === 'main' || model.type === 'vae';

/** A case's enqueue payload as Generate would build it: the graph and its batch data. */
export interface CalibrationGraph {
  caseId: string;
  graph: unknown;
  data: unknown;
}

/**
 * Builds every case's graph over `models`, or rejects naming the case that cannot compile, with the records' read
 * error, or once `signal` aborts. `models` are the records the server holds now, still being read when the compiler
 * is called, so it can fetch what else it needs meanwhile. Generation owns graph compilation and depends on Models,
 * so App supplies this; its implementation loads lazily on first call.
 */
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
  /** 1-based within the phase. */
  repetition: number;
}

/**
 * The plan's cases grouped by line (one decode dtype each), each case's warm-ups right before its measured runs.
 * Grouping keeps one precision switch per dtype: the first decode after a switch recasts the VAE, and the server
 * excludes those windows, so a switch must land on a warm-up.
 */
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

/**
 * Whether a plan read again when a calibration is confirmed asks for another calibration than the one confirmed: other
 * lines (an operation in another dtype), or another number of runs. The sizes follow the GPU memory free when the plan
 * is read, which a running generation moves by its working set, so new sizes alone are no reason to ask again.
 */
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

/**
 * The retry key of one step. Resending a step after a lost response with the same key returns the run the first
 * request created. The key carries the attempt id: the server keys its retry receipts on the key alone, so a key
 * reused by a new attempt would be refused.
 */
export const calibrationIdempotencyKey = (attemptId: string, step: CalibrationStep): string =>
  `webv2:wm-calibration:${attemptId}:${String(step.index)}:${step.case.id}:${step.phase}-${String(step.repetition)}`;

const byCreation = (left: WorkingMemoryRun, right: WorkingMemoryRun): number =>
  Date.parse(left.created_at) - Date.parse(right.created_at);

/** The id of the most recently created run, or null without runs. */
export const newestRunId = (runs: readonly WorkingMemoryRun[]): string | null =>
  [...runs].sort(byCreation).at(-1)?.run_id ?? null;

/**
 * The runs of one attempt, oldest first: those carrying `attemptId`, or by default those of the newest run's attempt.
 * A run without an attempt id (created before the server stored one) is an attempt of its own.
 */
export const selectAttemptRuns = (runs: readonly WorkingMemoryRun[], attemptId?: string): WorkingMemoryRun[] => {
  const ordered = [...runs].sort(byCreation);
  const newest = ordered.at(-1);
  const target = attemptId ?? newest?.attempt_id ?? null;

  if (target === null) {
    return newest ? [newest] : [];
  }

  return ordered.filter((run) => run.attempt_id === target);
};

/** What this tab knows of an attempt it started. */
export interface LocalAttempt {
  attemptId: string;
  /** Steps the attempt meant to queue. */
  total: number;
  /** Still posting steps. */
  submitting: boolean;
  /** Why posting stopped early, when it did. */
  submissionError: string | null;
  /** The user stopped it. */
  stopped: boolean;
  /**
   * The newest of the model's runs this tab knew when the attempt started. While the attempt has no runs of its own
   * (it failed before queuing anything), it is the latest attempt only until a newer run appears.
   */
  newestRunAtStart: string | null;
}

export type CalibrationActivity =
  | { kind: 'idle' }
  | {
      kind: 'running';
      /** Steps settled so far. */
      done: number;
      total: number;
      /** The case the queue is running or about to run, when it can be told. */
      currentCase: WorkingMemoryCase | null;
      /** Still posting steps. */
      queuing: boolean;
    };

/**
 * `discarded`: the attempt measured something, but the server aborted it (stopped, a run canceled, or its queuing
 * ended early), so none of its windows feeds a fit. A run that failed does not abort it: `kept` says that the other
 * steps measured something, which still counts.
 */
export type CalibrationOutcome =
  | { kind: 'none' }
  | { kind: 'completed' }
  /** Every run was canceled before it measured anything, so nothing was discarded. */
  | { kind: 'canceled' }
  /** `canceled`: runs were canceled, so the attempt did not stop because queuing ended early. */
  | { kind: 'partial'; canceled: boolean; discarded: boolean; error: string | null; queued: number; total: number }
  | { kind: 'failed'; discarded: boolean; error: string | null; kept: boolean };

/**
 * Progress and the latest attempt's outcome. A run counts as running while any of its items is pending or running
 * (the server derives that from the queue), so a pending item blocks a second start across reloads and tabs.
 */
export const deriveCalibrationProgress = ({
  local,
  runs,
  steps,
}: {
  local: LocalAttempt | null;
  runs: readonly WorkingMemoryRun[];
  steps: readonly CalibrationStep[];
}): { activity: CalibrationActivity; outcome: CalibrationOutcome } => {
  // Another tab may have started a newer attempt since; this tab's record then describes an older one. A record whose
  // attempt queued nothing (it failed to compile or to post its first step) stays current until a newer run exists.
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
  // The attempt's own step count: this tab's record, else the one the server stored with its runs, so any tab can tell
  // an attempt whose queuing ended early (a reload, a closed tab) from a complete one. A running attempt with neither
  // is assumed to follow the current plan; a settled one is judged by its runs alone, since the plan may have changed.
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
  // Once the attempt settled, the server alone says whether it discards the attempt's windows: a Stop that came after
  // the last step finished cancels nothing, and a step whose response was lost may still have been recorded. Before
  // the summary lists the attempt's runs, nothing of it was measured, so nothing reads as discarded either. Measured
  // means a run that completed, or partly did: `point_count` also counts the invalid windows of a decode that failed
  // or was canceled.
  const measured = attempt.some((run) => run.status === 'completed' || run.status === 'partial');
  const aborted = attempt.some((run) => run.held_back === 'aborted_attempt');
  const discarded = measured && aborted;
  const partial = (error: string | null): CalibrationOutcome => ({
    canceled,
    discarded,
    error,
    kind: 'partial',
    queued: attempt.length,
    total: plannedTotal,
  });

  // A submission error stands until the server lists every step this tab meant to queue: then the step whose answer
  // failed was recorded after all (a lost response, a 503 after queuing), and the runs alone tell the outcome.
  if (current?.submissionError && attempt.length < plannedTotal) {
    return {
      activity: idle,
      outcome:
        attempt.length === 0
          ? { discarded, error: current.submissionError, kept: false, kind: 'failed' }
          : partial(current.submissionError),
    };
  }

  if (attempt.length === 0) {
    return { activity: idle, outcome: current?.stopped ? { kind: 'canceled' } : { kind: 'none' } };
  }

  const failed = attempt.find((run) => run.status === 'failed');

  if (failed) {
    return {
      activity: idle,
      outcome: { discarded, error: failed.error, kept: measured && !aborted, kind: 'failed' },
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

/**
 * How a line reads: its server state, or `stopped` for a line that only stopped attempts measured. The server calls
 * that line `insufficient`, but its cause is the Stop, not a lack of sizes.
 */
export type LineDisplayState = WorkingMemoryLineState | 'stopped';

export const lineDisplayState = (line: Pick<WorkingMemoryLine, 'sizes' | 'state'>): LineDisplayState =>
  line.state === 'insufficient' &&
  line.sizes.length > 0 &&
  line.sizes.every((size) => size.excluded_reason === 'aborted_attempt')
    ? 'stopped'
    : line.state;

const LINE_STATE_SEVERITY: readonly LineDisplayState[] = ['unusable', 'stale', 'insufficient', 'stopped', 'in_use'];

/** The most severe of the lines' display states, or null without lines. */
export const summarizeLineStates = (states: readonly LineDisplayState[]): LineDisplayState | null =>
  LINE_STATE_SEVERITY.find((state) => states.includes(state)) ?? null;

/** What a summary says about the model as a whole. */
export interface CalibrationCoverage {
  /** The headline state: of the current environment's lines, or of all lines when it has none. */
  state: LineDisplayState | null;
  /** Some line of the current environment exists, measured by this model or another. */
  hasCurrentLines: boolean;
  /** This model has windows on some line or has runs, so a reset removes something. */
  hasOwnMeasurements: boolean;
  /** Estimates use lines that only other models measured. */
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
  // An estimator update leaves the old version's windows as an unusable line; once the new version's line applies,
  // the old one is superseded and does not head the section.
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

/**
 * Why Calibrate is unavailable, in the order a user can act on: nothing helps an unsupported model; a missing or
 * second GPU and an unusable store are machine problems; a running attempt only needs to finish.
 */
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

/**
 * Whether the server's way of measuring GPU memory was checked against real hardware on this GPU's backend: the
 * model summary's answer, else the current device class's; null while neither can tell.
 */
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

/** Changed facts the section names with a translated label. */
export type TranslatedFact = 'allocator' | 'driver' | 'invokeai' | 'osBuild' | 'sdpaGuard' | 'torchBuild';

/** Facts named by a product or runtime name, which needs no translation. */
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

/**
 * How the section names a changed environment fact (the server's fingerprint keys): a product name, a translated
 * label, or for environment variables and Invoke settings their own name. An unknown key shows as sent.
 */
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
