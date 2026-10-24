import { describe, expect, it } from 'vitest';

import {
  calibrationIdempotencyKey,
  changesCalibration,
  deriveCalibrationProgress,
  describeCoverage,
  describeFact,
  getCalibrationBlocker,
  isMeasurementMethodVerified,
  lineDisplayState,
  planCalibrationSteps,
  selectAttemptRuns,
  summarizeLineStates,
  type LocalAttempt,
  type WorkingMemoryCase,
  type WorkingMemoryLine,
  type WorkingMemoryLineSize,
  type WorkingMemoryPlan,
  type WorkingMemoryRun,
  type WorkingMemoryStatus,
} from './workingMemory';

const planCase = (dtype: string, side: number): WorkingMemoryCase => ({
  components: {},
  estimator: 'sd_vae_decode',
  height: side,
  id: `sd_vae_decode-decode-${dtype}-${String(side)}x${String(side)}`,
  line: `decode_${dtype}`,
  mode: 'txt2img',
  model_key: 'main',
  negative_prompt: '',
  positive_prompt: 'a lake',
  seed: 1,
  settings: { vaePrecision: dtype },
  steps: 2,
  width: side,
});

const plan = (
  cases: WorkingMemoryCase[],
  warmUp = 1,
  measured = 2
): Pick<WorkingMemoryPlan, 'cases' | 'repetitions'> => ({
  cases,
  repetitions: { measured, warm_up: warmUp },
});

const T0 = Date.parse('2026-10-01T12:00:00.000Z');

const run = (
  id: string,
  offsetSeconds: number,
  status: WorkingMemoryRun['status'],
  error: string | null = null,
  attemptId: string | null = 'attempt',
  overrides: Partial<WorkingMemoryRun> = {}
): WorkingMemoryRun => ({
  attempt_id: attemptId,
  attempt_steps: null,
  created_at: new Date(T0 + offsetSeconds * 1_000).toISOString(),
  error,
  held_back: null,
  // As the server counts: a failed or canceled decode leaves its window too, as an invalid one.
  point_count: status === 'running' ? 0 : 1,
  run_id: id,
  status,
  ...overrides,
});
const ABORTED = { held_back: 'aborted_attempt' } as const;

const describeSteps = (steps: ReturnType<typeof planCalibrationSteps>) =>
  steps.map((step) => `${step.case.line}@${String(step.case.width)}:${step.phase}${String(step.repetition)}`);

describe('planCalibrationSteps', () => {
  it('groups cases by precision and warms each case up right before measuring it', () => {
    // The plan lists an fp32 case between two fp16 ones; grouping keeps one precision switch.
    const steps = planCalibrationSteps(plan([planCase('fp16', 512), planCase('fp32', 512), planCase('fp16', 1024)]));

    expect(describeSteps(steps)).toEqual([
      'decode_fp16@512:warm_up1',
      'decode_fp16@512:measured1',
      'decode_fp16@512:measured2',
      'decode_fp16@1024:warm_up1',
      'decode_fp16@1024:measured1',
      'decode_fp16@1024:measured2',
      'decode_fp32@512:warm_up1',
      'decode_fp32@512:measured1',
      'decode_fp32@512:measured2',
    ]);
    expect(steps.map((step) => step.index)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('starts every precision with a warm-up, whatever the repetition counts', () => {
    const steps = planCalibrationSteps(
      plan([planCase('fp32', 768), planCase('fp16', 768), planCase('fp32', 1536)], 2, 3)
    );
    const switches = steps.filter((step, index) => index === 0 || steps[index - 1]!.case.line !== step.case.line);

    expect(steps).toHaveLength(15);
    expect(switches.map((step) => [step.case.line, step.phase])).toEqual([
      ['decode_fp32', 'warm_up'],
      ['decode_fp16', 'warm_up'],
    ]);
  });
});

describe('changesCalibration', () => {
  const confirmed = plan([planCase('fp16', 512), planCase('fp32', 512)]);

  it('lets new sizes through, which the free memory moves, but not other lines or another number of runs', () => {
    expect(changesCalibration(confirmed, plan([planCase('fp16', 768), planCase('fp32', 640)]))).toBe(false);
    // The same two cases in another order are the same calibration.
    expect(changesCalibration(confirmed, plan([planCase('fp32', 512), planCase('fp16', 512)]))).toBe(false);

    // fp32 is no longer measured, though the run count stays.
    expect(changesCalibration(confirmed, plan([planCase('fp16', 512), planCase('fp16', 768)]))).toBe(true);
    // A size more on the same lines.
    expect(
      changesCalibration(confirmed, plan([planCase('fp16', 512), planCase('fp16', 768), planCase('fp32', 512)]))
    ).toBe(true);
    // The same cases repeated more often.
    expect(changesCalibration(confirmed, plan([planCase('fp16', 512), planCase('fp32', 512)], 1, 3))).toBe(true);
  });
});

describe('calibrationIdempotencyKey', () => {
  const steps = planCalibrationSteps(plan([planCase('fp16', 512), planCase('fp32', 512)]));

  it('is the same for a resent step and differs across steps and attempts', () => {
    const keys = steps.map((step) => calibrationIdempotencyKey('attempt-a', step));

    expect(steps.map((step) => calibrationIdempotencyKey('attempt-a', step))).toEqual(keys);
    expect(new Set(keys).size).toBe(steps.length);
    expect(keys.some((key) => steps.some((step) => calibrationIdempotencyKey('attempt-b', step) === key))).toBe(false);
  });

  it('fits the server limit for plan ids of realistic length', () => {
    const longStep = planCalibrationSteps(
      plan([{ ...planCase('fp16', 4096), id: 'flux2_vae_decode-decode-bf16-4096x4096' }])
    )[2]!;

    expect(calibrationIdempotencyKey('0f8fad5b-d9cb-469f-a165-70867728950e', longStep).length).toBeLessThanOrEqual(255);
  });
});

describe('selectAttemptRuns', () => {
  it("takes the newest run's attempt, oldest run first, however long its runs took", () => {
    const runs = [
      run('new-2', 3_000, 'running', null, 'new'),
      run('old-1', 0, 'failed', null, 'old'),
      run('new-0', 300, 'completed', null, 'new'),
      run('old-2', 1, 'completed', null, 'old'),
    ];

    expect(selectAttemptRuns(runs).map((entry) => entry.run_id)).toEqual(['new-0', 'new-2']);
    expect(selectAttemptRuns(runs, 'old').map((entry) => entry.run_id)).toEqual(['old-1', 'old-2']);
    expect(selectAttemptRuns([])).toEqual([]);
  });

  it('counts each run without an attempt id as an attempt of its own', () => {
    const runs = [run('legacy-1', 0, 'completed', null, null), run('legacy-2', 1, 'failed', 'boom', null)];

    expect(selectAttemptRuns(runs).map((entry) => entry.run_id)).toEqual(['legacy-2']);
  });
});

describe('deriveCalibrationProgress', () => {
  const steps = planCalibrationSteps(plan([planCase('fp16', 512), planCase('fp32', 512)]));
  const local = (overrides: Partial<LocalAttempt> = {}): LocalAttempt => ({
    attemptId: 'mine',
    newestRunAtStart: null,
    stopped: false,
    submissionError: null,
    submitting: false,
    total: steps.length,
    ...overrides,
  });

  it('reports progress and the case the queue is on while any run is pending or running', () => {
    const runs = [
      run('r0', 0, 'completed'),
      run('r1', 1, 'completed'),
      run('r2', 2, 'running'),
      run('r3', 3, 'running'),
    ];

    expect(deriveCalibrationProgress({ local: null, runs, steps })).toEqual({
      activity: { currentCase: steps[2]!.case, done: 2, kind: 'running', queuing: false, total: 6 },
      outcome: { kind: 'none' },
    });
  });

  it('counts an attempt that is still queuing as running before the server lists any run', () => {
    const { activity } = deriveCalibrationProgress({ local: local({ submitting: true }), runs: [], steps });

    expect(activity).toEqual({ currentCase: steps[0]!.case, done: 0, kind: 'running', queuing: true, total: 6 });
  });

  it('reports the error of a failed run, which discards the attempt only where the attempt was aborted too', () => {
    // An out-of-memory step ends its run, not the attempt: the other steps still feed the fit.
    const ranOn = [run('r0', 0, 'completed'), run('r1', 1, 'failed', 'CUDA out of memory'), run('r2', 2, 'completed')];
    // Stopped after the failure: the server holds the attempt back as aborted.
    const stopped = [
      run('r0', 0, 'completed', null, 'attempt', ABORTED),
      run('r1', 1, 'failed', 'CUDA out of memory', 'attempt', ABORTED),
      run('r2', 2, 'canceled', null, 'attempt', ABORTED),
    ];

    expect(deriveCalibrationProgress({ local: null, runs: ranOn, steps }).outcome).toEqual({
      discarded: false,
      error: 'CUDA out of memory',
      kept: true,
      kind: 'failed',
    });
    expect(deriveCalibrationProgress({ local: null, runs: stopped, steps }).outcome).toEqual({
      discarded: true,
      error: 'CUDA out of memory',
      kept: false,
      kind: 'failed',
    });
    // Every step failed out of memory: each left an invalid window, and nothing still counts.
    const allFailed = steps.map((_, index) => run(`f${String(index)}`, index, 'failed', 'OOM'));
    expect(deriveCalibrationProgress({ local: null, runs: allFailed, steps }).outcome).toEqual({
      discarded: false,
      error: 'OOM',
      kept: false,
      kind: 'failed',
    });
    // The first step failed and Stop canceled the rest: the server holds the attempt back, but its only window is
    // invalid, so nothing was discarded.
    const stoppedMidDecode = [
      run('f0', 0, 'failed', 'OOM', 'attempt', ABORTED),
      ...steps
        .slice(1)
        .map((_, index) => run(`c${String(index + 1)}`, index + 1, 'canceled', null, 'attempt', ABORTED)),
    ];
    expect(deriveCalibrationProgress({ local: null, runs: stoppedMidDecode, steps }).outcome).toEqual({
      discarded: false,
      error: 'OOM',
      kept: false,
      kind: 'failed',
    });
  });

  it('tells completed, canceled and partial attempts apart', () => {
    const completed = steps.map((_, index) => run(`r${String(index)}`, index, 'completed'));
    // A queue clear canceled these: the server discards the attempt.
    const canceled = [
      run('r0', 0, 'canceled', null, 'attempt', ABORTED),
      run('r1', 1, 'canceled', null, 'attempt', ABORTED),
    ];
    const someCanceled = [
      ...completed.slice(0, 3).map((settled) => ({ ...settled, ...ABORTED })),
      run('r3', 3, 'canceled', null, 'attempt', ABORTED),
      run('r4', 4, 'partial', null, 'attempt', ABORTED),
    ];

    expect(deriveCalibrationProgress({ local: null, runs: completed, steps }).outcome).toEqual({ kind: 'completed' });
    // Every run was canceled, the first one during its decode: its window is invalid, so nothing was lost.
    expect(deriveCalibrationProgress({ local: null, runs: canceled, steps }).outcome).toEqual({ kind: 'canceled' });
    // A canceled run of no attempt (a script's) is judged on its own.
    expect(
      deriveCalibrationProgress({ local: null, runs: [run('r0', 0, 'canceled', null, null)], steps }).outcome
    ).toEqual({ kind: 'canceled' });
    expect(deriveCalibrationProgress({ local: null, runs: someCanceled, steps }).outcome).toEqual({
      canceled: true,
      discarded: true,
      error: null,
      kind: 'partial',
      queued: 5,
      total: 5,
    });
  });

  it('judges an attempt another tab started by its runs, not by a plan that may have changed since', () => {
    const fewerThanPlanned = [run('r0', 0, 'completed'), run('r1', 1, 'completed')];

    expect(deriveCalibrationProgress({ local: null, runs: fewerThanPlanned, steps }).outcome).toEqual({
      kind: 'completed',
    });
  });

  it('reports an attempt whose queuing stopped early as partial, or failed when nothing was queued', () => {
    // Two of six steps were queued: the server holds the attempt back as aborted.
    const shortOf = { ...ABORTED, attempt_steps: 6 };
    const runs = [run('r0', 0, 'completed', null, 'mine', shortOf), run('r1', 1, 'completed', null, 'mine', shortOf)];

    expect(deriveCalibrationProgress({ local: local({ submissionError: 'Queue full' }), runs, steps }).outcome).toEqual(
      {
        canceled: false,
        discarded: true,
        error: 'Queue full',
        kind: 'partial',
        queued: 2,
        total: 6,
      }
    );
    // Its graphs did not compile, so nothing of it was queued; an older attempt does not hide the error.
    const older = [run('old', 0, 'completed', null, 'old')];
    const failedStart = local({ newestRunAtStart: 'old', submissionError: 'Case x: unknown model' });
    expect(deriveCalibrationProgress({ local: failedStart, runs: older, steps }).outcome).toEqual({
      discarded: false,
      error: 'Case x: unknown model',
      kept: false,
      kind: 'failed',
    });
  });

  it('follows the server once the attempt settled, whatever this tab saw of its end', () => {
    const recorded = { attempt_steps: 6 };
    const finished = steps.map((_, index) => run(`r${String(index)}`, index, 'completed', null, 'mine', recorded));

    // Stop came after the last step finished: nothing was canceled, and the server keeps the attempt.
    expect(deriveCalibrationProgress({ local: local({ stopped: true }), runs: finished, steps }).outcome).toEqual({
      kind: 'completed',
    });
    // Stop came while the last step failed: the server keeps the other steps.
    const lastFailed = [...finished.slice(0, 5), run('r5', 5, 'failed', 'CUDA out of memory', 'mine', recorded)];
    expect(deriveCalibrationProgress({ local: local({ stopped: true }), runs: lastFailed, steps }).outcome).toEqual({
      discarded: false,
      error: 'CUDA out of memory',
      kept: true,
      kind: 'failed',
    });
    // The last step's answer failed three times, but the server had recorded it: the attempt is complete.
    expect(
      deriveCalibrationProgress({
        local: local({ submissionError: 'The calibration store is locked' }),
        runs: finished,
        steps,
      }).outcome
    ).toEqual({ kind: 'completed' });
  });

  it('lets an attempt that queued nothing give way to any attempt that ran after it', () => {
    // This tab could not reach the server; another tab then calibrated to the end.
    const failedStart = local({ newestRunAtStart: 'old', submissionError: 'Failed to fetch' });
    const theirs = steps.map((_, index) => run(`theirs-${String(index)}`, 100 + index, 'completed', null, 'theirs'));

    expect(
      deriveCalibrationProgress({
        local: failedStart,
        runs: [run('old', 0, 'completed', null, 'old'), ...theirs],
        steps,
      }).outcome
    ).toEqual({ kind: 'completed' });
    expect(
      deriveCalibrationProgress({ local: { ...failedStart, newestRunAtStart: null }, runs: theirs, steps }).outcome
    ).toEqual({ kind: 'completed' });
  });

  it('reports an attempt the user stopped as canceled, even where queuing ended before any run was canceled', () => {
    // Stopped after two of six steps were queued: the server holds the attempt back as aborted.
    const shortOf = { ...ABORTED, attempt_steps: 6 };
    const runs = [run('r0', 0, 'completed', null, 'mine', shortOf), run('r1', 1, 'completed', null, 'mine', shortOf)];

    // Stop discards the attempt, the steps it finished included, before the server's next summary says so.
    expect(deriveCalibrationProgress({ local: local({ stopped: true }), runs, steps }).outcome).toEqual({
      canceled: true,
      discarded: true,
      error: null,
      kind: 'partial',
      queued: 2,
      total: 6,
    });
    // Stopped before its first step was queued: nothing was measured to discard.
    expect(deriveCalibrationProgress({ local: local({ stopped: true }), runs: [], steps }).outcome).toEqual({
      kind: 'canceled',
    });
  });

  it('tells an attempt whose queuing ended early from a complete one by the step count stored with its runs', () => {
    const counted = { attempt_steps: 6 };
    const truncated = [
      run('r0', 0, 'completed', null, 'attempt', { ...counted, ...ABORTED }),
      run('r1', 1, 'completed', null, 'attempt', { ...counted, ...ABORTED }),
    ];
    const complete = steps.map((_, index) => run(`r${String(index)}`, index, 'completed', null, 'attempt', counted));

    // After a reload this tab has no record of it.
    expect(deriveCalibrationProgress({ local: null, runs: truncated, steps }).outcome).toEqual({
      canceled: false,
      discarded: true,
      error: null,
      kind: 'partial',
      queued: 2,
      total: 6,
    });
    expect(deriveCalibrationProgress({ local: null, runs: complete, steps }).outcome).toEqual({ kind: 'completed' });
  });

  it('follows the newer attempt another tab started, not this tab’s older one', () => {
    const runs = [
      run('mine', 0, 'failed', 'old failure', 'mine'),
      run('theirs-0', 100, 'completed', null, 'theirs'),
      run('theirs-1', 101, 'running', null, 'theirs'),
    ];
    const { activity } = deriveCalibrationProgress({ local: local(), runs, steps });

    expect(activity).toMatchObject({ done: 1, kind: 'running' });
  });
});

describe('summarizeLineStates', () => {
  it('reports the most severe line, or null without lines', () => {
    expect(summarizeLineStates(['in_use', 'stale'])).toBe('stale');
    expect(summarizeLineStates(['insufficient', 'in_use'])).toBe('insufficient');
    expect(summarizeLineStates(['stopped', 'in_use'])).toBe('stopped');
    expect(summarizeLineStates(['stopped', 'insufficient'])).toBe('insufficient');
    expect(summarizeLineStates(['stale', 'unusable'])).toBe('unusable');
    expect(summarizeLineStates(['in_use'])).toBe('in_use');
    expect(summarizeLineStates([])).toBeNull();
  });
});

describe('lineDisplayState', () => {
  const size = (excludedReason: WorkingMemoryLineSize['excluded_reason']) => ({ excluded_reason: excludedReason });
  const sized = (state: WorkingMemoryLine['state'], reasons: WorkingMemoryLineSize['excluded_reason'][]) =>
    ({ sizes: reasons.map(size), state }) as Pick<WorkingMemoryLine, 'sizes' | 'state'>;

  it('reads a line only stopped attempts measured as stopped, not as short of sizes', () => {
    expect(lineDisplayState(sized('insufficient', ['aborted_attempt', 'aborted_attempt']))).toBe('stopped');
    // A size of its own, or one left out for another reason, makes it short of sizes after all.
    expect(lineDisplayState(sized('insufficient', ['aborted_attempt', null]))).toBe('insufficient');
    expect(lineDisplayState(sized('insufficient', ['aborted_attempt', 'fragmented']))).toBe('insufficient');
    expect(lineDisplayState(sized('insufficient', []))).toBe('insufficient');
    expect(lineDisplayState(sized('in_use', ['aborted_attempt', null, null]))).toBe('in_use');
  });
});

describe('describeCoverage', () => {
  type Line = Pick<WorkingMemoryLine, 'env_id' | 'estimator' | 'line' | 'own_windows' | 'sizes' | 'state' | 'version'>;
  const line = (
    state: WorkingMemoryLine['state'],
    {
      envId = 'current',
      lineName = 'decode_fp16',
      ownWindows = 3,
      version = 2,
    }: { envId?: string; lineName?: string; ownWindows?: number; version?: number } = {}
  ): Line => ({
    env_id: envId,
    estimator: 'sd_vae_decode',
    line: lineName,
    own_windows: ownWindows,
    sizes: [],
    state,
    version,
  });
  const coverage = (lines: Line[], runs: WorkingMemoryRun[] = []) =>
    describeCoverage({ current_env_id: 'current', lines, runs });

  it("takes the headline from the current environment's lines, else from all lines", () => {
    // An old environment's unusable line does not outrank what applies now.
    expect(coverage([line('unusable', { envId: 'old' }), line('in_use')])).toMatchObject({
      hasCurrentLines: true,
      state: 'in_use',
    });
    expect(coverage([line('unusable', { envId: 'old' })])).toMatchObject({ hasCurrentLines: false, state: 'unusable' });
    expect(coverage([])).toMatchObject({ hasCurrentLines: false, hasOwnMeasurements: false, state: null });
  });

  it('lets a line recalibrated under a new estimator version supersede the old version’s unusable line', () => {
    const oldVersion = line('unusable', { version: 1 });

    expect(coverage([oldVersion, line('in_use')]).state).toBe('in_use');
    expect(coverage([oldVersion, line('stale')]).state).toBe('stale');
    expect(coverage([oldVersion]).state).toBe('unusable');
    // Only the same line supersedes it.
    expect(coverage([oldVersion, line('in_use', { lineName: 'decode_fp32' })]).state).toBe('unusable');
    expect(coverage([oldVersion, line('insufficient')]).state).toBe('unusable');
  });

  it('tells lines only other models measured from the model’s own', () => {
    expect(coverage([line('in_use', { ownWindows: 0 })])).toMatchObject({
      hasOwnMeasurements: false,
      usesOthersOnly: true,
    });
    expect(coverage([line('in_use', { ownWindows: 0 }), line('stale')])).toMatchObject({
      hasOwnMeasurements: true,
      usesOthersOnly: false,
    });
    // A shared line that does not apply is not used.
    expect(coverage([line('insufficient', { ownWindows: 0 })]).usesOthersOnly).toBe(false);
    // Runs that measured nothing still leave something to reset.
    expect(coverage([], [run('r0', 0, 'canceled')]).hasOwnMeasurements).toBe(true);
  });
});

describe('getCalibrationBlocker', () => {
  const supported = { reason: null, supported: true };
  const available = { available: true, unavailable_reason: null };
  const usable = { problem: null, usable: true };

  it('names the reason a user can least act on first', () => {
    expect(
      getCalibrationBlocker({
        isRunning: true,
        plan: { reason: 'Not a text-to-image model.', supported: false },
        status: { available: false, unavailable_reason: 'Two GPUs.' },
        summary: null,
      })
    ).toEqual({ kind: 'unsupported', reason: 'Not a text-to-image model.' });
    expect(
      getCalibrationBlocker({
        isRunning: true,
        plan: supported,
        status: { available: false, unavailable_reason: 'File unusable.' },
        summary: { problem: 'schema_version 9', usable: false },
      })
    ).toEqual({ kind: 'storeUnusable', reason: 'schema_version 9' });
    expect(
      getCalibrationBlocker({
        isRunning: false,
        plan: supported,
        status: { available: false, unavailable_reason: 'Two GPUs.' },
        summary: null,
      })
    ).toEqual({ kind: 'unavailable', reason: 'Two GPUs.' });
    expect(getCalibrationBlocker({ isRunning: true, plan: supported, status: available, summary: usable })).toEqual({
      kind: 'running',
    });
    expect(getCalibrationBlocker({ isRunning: false, plan: supported, status: available, summary: usable })).toBeNull();
  });
});

describe('isMeasurementMethodVerified', () => {
  const status = (verified: boolean | null): WorkingMemoryStatus => ({
    available: true,
    current_device_class_id: 'class',
    current_env_id: 'env',
    device_classes: [
      {
        device_class: { arch: null, backend: 'cuda', id: 'class', name: 'GPU', total_mib: 8192, vendor: 'x' },
        device_class_id: 'class',
        envs: [],
        is_current: true,
        measurement_method: 'allocator',
        measurement_verified: verified,
        point_counts: [],
        problem: null,
        runs: [],
        usable: true,
      },
    ],
    unavailable_reason: null,
  });

  it("takes the model summary's answer, then the current device class's", () => {
    expect(isMeasurementMethodVerified({ measurement_verified: false }, status(true))).toBe(false);
    expect(isMeasurementMethodVerified({ measurement_verified: null }, status(false))).toBe(false);
    expect(isMeasurementMethodVerified(null, status(true))).toBe(true);
    expect(isMeasurementMethodVerified(null, status(null))).toBeNull();
  });
});

describe('describeFact', () => {
  it('names runtimes by product, translates the rest, and shows variables and settings by their own name', () => {
    expect(['torch', 'hip', 'cudnn', 'os'].map(describeFact)).toEqual([
      { name: 'PyTorch' },
      { name: 'ROCm' },
      { name: 'cuDNN' },
      { name: 'OS' },
    ]);
    expect(['driver', 'allocator_backend', 'rocm_sdpa_guard'].map(describeFact)).toEqual([
      { translated: 'driver' },
      { translated: 'allocator' },
      { translated: 'sdpaGuard' },
    ]);
    expect(describeFact('setting.attention_type')).toEqual({ name: 'attention_type' });
    expect(describeFact('env.PYTORCH_CUDA_ALLOC_CONF')).toEqual({ name: 'PYTORCH_CUDA_ALLOC_CONF' });
    expect(describeFact('something_new')).toEqual({ name: 'something_new' });
  });
});
