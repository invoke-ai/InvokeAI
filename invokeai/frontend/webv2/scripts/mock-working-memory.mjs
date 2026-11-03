/**
 * Working-memory calibration for the mock backend: plans, summaries, the device status, calibration runs and resets
 * (`/api/v2/models/.../working_memory`), and cancelling a run's items by origin. The mock has no Socket.IO, so a run
 * advances when a client reads the model's summary: a read at least a second after the last step finishes the running
 * item and starts the next, which a section polling every two seconds turns into visible progress.
 *
 * Main models (`fixture-model-001`, `-006`, ...) each show one state, by key (`MODEL_SCENARIOS`); every other main model is never
 * calibrated, and VAEs are unsupported. `SHARED_LINES` pairs models whose decode lines pool their measurements, so one
 * of them shows lines only the other measured. `fixture-model-046` follows a warm RX 9060 XT: its largest fp32 size
 * measured allocator fragmentation and is not counted, and two of three windows at one fp16 size ran partially resident,
 * which leaves that size counted on the third. `POST /__faults` with `workingMemory` selects a device scenario instead,
 * with `stoppedCalibration: <model key>` replaces that main model's runs with an attempt Stop aborted, with
 * `runningCalibration: <model key>` with one still running, which holds the GPU, and with
 * `modelChangedCalibration: <model key>` with one whose later steps failed because the model changed after queuing.
 * A calibrate request with an attempt is refused (409, coded) while another attempt runs on the GPU or is open (between
 * its steps, short of them, not aborted, active within the last two minutes of `clock`), for a later step once another
 * run was created since the attempt's first (`calibration_interleaved`), and for a graph naming a model whose hash,
 * base, type, format or variant changed since it compiled, or that was removed. Like the server, it takes a request only
 * with `graph_models` stating every model the graph names and no other (422).
 * Calibration items live here rather than in the fixture queue, so profile counts stay fixed. Tests pass their own
 * `clock` to advance runs without waiting.
 */

export const WORKING_MEMORY_SCENARIOS = new Set(['ok', 'multiGpu', 'noGpu', 'storeUnusable', 'error', 'notVerified']);

const ORIGIN_PREFIX = 'invokeai:wm-calibration:';
const ATTEMPT_ID_PATTERN = /^[A-Za-z0-9_.:-]{1,64}$/;
const DEVICE_CLASS = {
  arch: 'sm_89',
  backend: 'cuda',
  id: '0123456789abcdef',
  name: 'NVIDIA GeForce RTX 4090',
  total_mib: 24_564,
  vendor: 'nvidia',
};
const ENV_ID = 'fedcba9876543210';
const OLD_ENV_ID = '00ff00ff00ff00ff';
const MIB = 2 ** 20;
const SIDES = [768, 1088, 1536];
const DTYPES = ['fp16', 'fp32'];
const REPETITIONS = { measured: 2, warm_up: 1 };
const ENCODE_NOT_COVERED =
  'encode: text-to-image runs no encode, so image-to-image and inpainting encodes keep the shipped values';
const SEEDED_AT = Date.parse('2025-12-01T10:00:00.000Z');
/** The server's problem for a line that only aborted attempts measured. */
const ABORTED_LINE_PROBLEM =
  'Stop or an interrupted submission discarded the measurements of this line: its windows come from aborted ' +
  'attempts, which never count. Calibrate again to measure it.';
/** The server's bounds on an attempt's step count. */
const MAX_ATTEMPT_STEPS = 256;
/** A run's items advance at most this often, so a section that refetches on every queued step still shows progress. */
const STEP_MS = 1_000;

/** What each seeded main model shows, by key. */
const MODEL_SCENARIOS = {
  'fixture-model-006': 'completed',
  'fixture-model-011': 'stale',
  'fixture-model-016': 'unusable',
  'fixture-model-026': 'failed',
  'fixture-model-031': 'unsupported',
  'fixture-model-036': 'insufficient',
  'fixture-model-046': 'fragmented',
};

/** The sizes the `fragmented` scenario measured, as the 9060 XT plan picked them. */
const FRAGMENTED_SIDES = [576, 832, 1216];

/** Why a seeded window of the `fragmented` scenario is excluded from the fit, by its step; undefined when it counts. */
const excludedIn = (scenario, step, repetition) => {
  if (scenario !== 'fragmented') {
    return undefined;
  }

  if (step.line === 'decode_fp32' && step.width === 1216) {
    return 'fragmented';
  }

  return step.line === 'decode_fp16' && step.width === 832 && repetition < 2 ? 'partial_residency' : undefined;
};

/** Models whose lines pool both models' measurements; the first is never calibrated itself. */
const SHARED_LINES = [['fixture-model-041', 'fixture-model-006']];

const partnersOf = (modelKey) =>
  SHARED_LINES.flatMap(([left, right]) => (left === modelKey ? [right] : right === modelKey ? [left] : []));

/** A plausible decode demand: the VAE's weights plus a per-pixel activation cost, twice that in fp32. */
const demandOf = (pixels, dtype) => Math.round((dtype === 'fp32' ? 2 : 1) * (420 * MIB + pixels * 1_100));

const casesOf = (model) =>
  DTYPES.flatMap((dtype) =>
    (MODEL_SCENARIOS[model.key] === 'fragmented' ? FRAGMENTED_SIDES : SIDES).map((side) => ({
      components: {},
      estimator: 'sd_vae_decode',
      height: side,
      id: `sd_vae_decode-decode-${dtype}-${side}x${side}`,
      line: `decode_${dtype}`,
      mode: 'txt2img',
      model_key: model.key,
      negative_prompt: '',
      positive_prompt: 'a mountain lake at dawn, detailed photograph',
      seed: 1,
      settings: { vaePrecision: dtype },
      steps: 2,
      width: side,
    }))
  );

const stepsOf = (cases) =>
  DTYPES.flatMap((dtype) =>
    cases
      .filter((item) => item.line === `decode_${dtype}`)
      .flatMap((item) => Array.from({ length: REPETITIONS.warm_up + REPETITIONS.measured }, () => item))
  );

/** The server's refusal while another attempt runs on the GPU: one attempt per device class at a time. */
const RUNNING_REFUSAL = {
  code: 'calibration_running',
  message:
    'Another calibration is still running on this GPU. Wait until it has finished, or stop it, then calibrate again.',
};

/** The server's refusal of a later step once another calibration ran on the GPU since the attempt began. */
const INTERLEAVED_REFUSAL = {
  code: 'calibration_interleaved',
  message:
    'Another calibration ran on this GPU after this calibration started, so its measurements can no longer be combined. Calibrate again.',
};

/** How long an attempt between its steps holds the GPU after its last activity, as the server waits. */
const ATTEMPT_IDLE_GRACE_MS = 120_000;

/** As webv2's `compiledModelIdentities` reads one: string key, hash, name, base and type; no submodel type but a string. */
const isModelIdentifier = (value) =>
  ['key', 'hash', 'name', 'base', 'type'].every((field) => typeof value[field] === 'string') &&
  (value.submodel_type === undefined || value.submodel_type === null || typeof value.submodel_type === 'string');

/** Every fact of a `graph_models` entry is stated; these three are never null, as the server requires. */
const isStatedIdentity = (identity) =>
  identity !== null &&
  typeof identity === 'object' &&
  IDENTITY_FACTS.every((fact) => fact in identity) &&
  ['hash', 'base', 'type'].every((fact) => typeof identity[fact] === 'string');

/** The model identifiers a graph names, as the server finds them: in the nodes' fields and nested in them. */
const modelIdentifiersOf = (graph) => {
  const found = [];
  const visit = (value) => {
    if (Array.isArray(value)) {
      value.forEach(visit);
    } else if (value !== null && typeof value === 'object') {
      if (isModelIdentifier(value)) {
        found.push(value);
      } else {
        Object.values(value).forEach(visit);
      }
    }
  };

  for (const node of Object.values(graph.nodes ?? {})) {
    if (node !== null && typeof node === 'object') {
      Object.values(node).forEach(visit);
    }
  }
  return found;
};

const IDENTITY_FACTS = ['hash', 'base', 'type', 'format', 'variant'];

/** Why `graphModels` does not state exactly the graph's models, every fact of each, or null when it does. */
const graphModelsProblem = (graph, graphModels) => {
  if (graphModels === null || typeof graphModels !== 'object' || Array.isArray(graphModels)) {
    return 'graph_models is required: the identity of every model the graph names.';
  }

  const named = new Set(modelIdentifiersOf(graph).map((identifier) => identifier.key));
  const stated = Object.keys(graphModels);
  const problems = [
    ...[...named]
      .filter((key) => !(key in graphModels))
      .map((key) => `the graph names ${key}, which graph_models lacks`),
    ...stated.filter((key) => !named.has(key)).map((key) => `graph_models names ${key}, which the graph does not`),
    ...stated
      .filter((key) => !isStatedIdentity(graphModels[key]))
      .map((key) => `graph_models[${key}] does not state every fact, or states a null hash, base or type`),
  ];

  return problems.length > 0
    ? `graph_models must state every model the graph names, and only those: ${problems.join('; ')}`
    : null;
};

/**
 * `currentModel` reads a model's record as it is now, so a graph compiled over an older record is refused as the
 * server refuses it; the fixture's records stand in without it.
 */
export const createWorkingMemoryMock = (models, { clock = Date.now, currentModel } = {}) => {
  const mainModels = models.filter((model) => model.type === 'main');
  const byKey = new Map(models.map((model) => [model.key, model]));
  /** Runs by model key, oldest first: `{ run_id, created_at, items: [{ item_id, case, status, error }] }`. */
  const runs = new Map();
  /** Measured points by model key: `{ line, width, height, bytes, env_id }`. */
  const points = new Map();
  /** Run ids by idempotency key, so a resent request returns the run it created. */
  const receipts = new Map();
  let nextRunNumber = 1;
  let nextItemId = 900_000;
  let createdCount = 0;
  let lastAdvanceAt = 0;

  const now = () => new Date(SEEDED_AT + 86_400_000 * 30 + createdCount++ * 1_000).toISOString();

  /**
   * `excluded` names why the fit leaves the window out: `fragmented` or `partial_residency`. A window of an aborted
   * attempt is left out as well, judged when lines are read (`heldBack`).
   */
  const addPoint = (modelKey, run, item, envId = ENV_ID, excluded = undefined) => {
    const dtype = item.case.line.split('_').at(-1);
    const list = points.get(modelKey) ?? [];
    const demand = demandOf(item.case.width * item.case.height, dtype);

    list.push({
      // A fragmented allocator reserves well beyond what the decode needs.
      bytes: excluded === 'fragmented' ? Math.round(demand * 1.4) : demand,
      env_id: envId,
      excluded,
      height: item.case.height,
      line: item.case.line,
      run_id: run.run_id,
      width: item.case.width,
    });
    points.set(modelKey, list);
  };

  const createRun = (modelKey, steps, createdAt, attemptId, attemptSteps) => {
    const run = {
      attempt_id: attemptId,
      attempt_steps: attemptSteps,
      created_at: createdAt,
      items: steps.map((step) => ({ case: step, error: null, item_id: nextItemId++, status: 'pending' })),
      // When the run last did something, on `clock`: its creation, then each item that settles.
      lastActivityAt: clock(),
      run_id: `mock-wm-run-${String(nextRunNumber++).padStart(4, '0')}`,
    };
    const list = runs.get(modelKey) ?? [];

    list.push(run);
    runs.set(modelKey, list);

    return run;
  };

  /** Seeds one model's state: settled runs one per step, two seconds apart, a month before now. */
  const seed = (model, scenario) => {
    const cases = casesOf(model);
    const steps = scenario === 'insufficient' ? stepsOf(cases).slice(0, 3) : stepsOf(cases);
    const settledRuns = steps.map((step, index) =>
      createRun(model.key, [step], new Date(SEEDED_AT + index * 2_000).toISOString(), `seed-${model.key}`, steps.length)
    );

    settledRuns.forEach((run, index) => {
      const [item] = run.items;
      const isRunning = scenario === 'running' && index >= 4;

      // Seeded a month ago: a short seeded attempt is long abandoned, never open.
      run.lastActivityAt = Number.NEGATIVE_INFINITY;

      if (scenario === 'failed' && index === 7) {
        item.status = 'failed';
        item.error = 'OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB';
      } else if (scenario === 'stopped' && index >= 4) {
        // Stop canceled what the queue still held: the attempt is aborted, the four finished steps included.
        item.status = 'canceled';
      } else if (scenario === 'modelChanged' && index >= 4) {
        // The record changed after queuing: the server failed the later items as it admitted them.
        item.status = 'failed';
        item.errorCode = 'model_changed';
        item.error = `${model.name} changed after this calibration was planned (hash). Calibrate again.`;
      } else if (isRunning) {
        item.status = index === 4 ? 'in_progress' : 'pending';
      } else {
        const repetition = index % (REPETITIONS.warm_up + REPETITIONS.measured);

        item.status = 'completed';
        addPoint(
          model.key,
          run,
          item,
          scenario === 'unusable' ? OLD_ENV_ID : ENV_ID,
          excludedIn(scenario, item.case, repetition)
        );
      }
    });
  };

  const reset = () => {
    runs.clear();
    points.clear();
    receipts.clear();
    nextRunNumber = 1;
    nextItemId = 900_000;
    createdCount = 0;
    lastAdvanceAt = 0;

    for (const model of mainModels) {
      const scenario = MODEL_SCENARIOS[model.key];

      if (scenario && scenario !== 'unsupported') {
        seed(model, scenario);
      }
    }
  };

  reset();

  /**
   * Replaces a main model's runs with an attempt stopped after four of its steps, so a browser check can review that
   * state (`POST /__faults` with `stoppedCalibration`); false for a key that names no main model. `reset` drops it.
   */
  const seedStopped = (modelKey) => seedOpted(modelKey, 'stopped');

  /**
   * Replaces a main model's runs with an attempt still running, which holds the GPU (`POST /__faults` with
   * `runningCalibration`); false for a key that names no main model.
   */
  const seedRunning = (modelKey) => seedOpted(modelKey, 'running');

  /**
   * Replaces a main model's runs with an attempt whose later steps failed because the model's record changed after
   * they were queued (`POST /__faults` with `modelChangedCalibration`); false for a key that names no main model.
   */
  const seedModelChanged = (modelKey) => seedOpted(modelKey, 'modelChanged');

  const seedOpted = (modelKey, scenario) => {
    const model = mainModels.find((candidate) => candidate.key === modelKey);

    if (!model) {
      return false;
    }

    runs.delete(model.key);
    points.delete(model.key);
    seed(model, scenario);
    return true;
  };

  const runStatus = (run, pointCount) => {
    const statuses = run.items.map((item) => item.status);

    if (statuses.some((status) => status === 'pending' || status === 'in_progress')) {
      return { error: null, error_code: null, status: 'running' };
    }

    const failed = run.items.find((item) => item.status === 'failed');

    if (failed) {
      return { error: failed.error, error_code: failed.errorCode ?? null, status: 'failed' };
    }

    if (statuses.every((status) => status === 'completed')) {
      return { error: null, error_code: null, status: 'completed' };
    }

    return { error: null, error_code: null, status: pointCount > 0 ? 'partial' : 'canceled' };
  };

  /** The windows that count: those of the run's completed items. */
  const pointCountOf = (run) => run.items.filter((item) => item.status === 'completed').length;
  /**
   * The windows the run stored, as `point_count` reports them: the valid ones, plus the invalid window a decode
   * leaves that failed or was canceled while it ran.
   */
  const storedPointsOf = (run) =>
    pointCountOf(run) + run.items.filter((item) => item.status === 'failed' || item.interrupted).length;

  /**
   * Why no window of a run feeds a fit, by run id, judged per attempt as the server does: `running` while a run of it
   * is unsettled; `aborted_attempt` once none is and a run of it ended canceled (Stop, a queue clear), failed because
   * the model changed after it was planned, or it holds fewer runs than it planned (Stop, a failed submission). Any
   * other failed run ends that run, not the attempt; a run without an attempt is never aborted.
   */
  const statusesOf = (attempt) => attempt.map((run) => runStatus(run, pointCountOf(run)).status);
  const isShort = (attempt) => attempt.length < Math.max(...attempt.map((run) => run.attempt_steps ?? 0));
  /** A run of the attempt ended canceled (Stop, a queue clear) or failed because the model changed. */
  const endedEarly = (attempt) =>
    statusesOf(attempt).some((status) => status === 'canceled' || status === 'partial') ||
    attempt.some((run) => run.items.some((item) => item.status === 'failed' && item.errorCode === 'model_changed'));
  /**
   * Whether another attempt than `attemptId` holds the GPU, as the server's `_attempt_blocks` judges it: a run of it is
   * running, or it is open: between its steps, short of them, nothing ended it early, and active within the grace.
   */
  const attemptBlocks = (attemptId) =>
    [
      ...Map.groupBy(
        [...runs.values()].flat().filter((run) => run.attempt_id !== null && run.attempt_id !== attemptId),
        (run) => run.attempt_id
      ).values(),
    ].some(
      (attempt) =>
        statusesOf(attempt).includes('running') ||
        (isShort(attempt) &&
          !endedEarly(attempt) &&
          clock() - Math.max(...attempt.map((run) => run.lastActivityAt)) <= ATTEMPT_IDLE_GRACE_MS)
    );

  const heldBack = () => {
    const reasons = new Map();

    for (const list of runs.values()) {
      const attempts = Map.groupBy(list, (run) => run.attempt_id ?? `run:${run.run_id}`);

      for (const attempt of attempts.values()) {
        const aborted = attempt[0].attempt_id !== null && (isShort(attempt) || endedEarly(attempt));
        const reason = statusesOf(attempt).includes('running') ? 'running' : aborted ? 'aborted_attempt' : null;

        for (const run of attempt) {
          reasons.set(run.run_id, reason);
        }
      }
    }

    return reasons;
  };

  const modelRuns = (modelKey) => {
    const reasons = heldBack();

    return (runs.get(modelKey) ?? []).map((run) => {
      const pointCount = pointCountOf(run);

      return {
        attempt_id: run.attempt_id,
        attempt_steps: run.attempt_steps,
        created_at: run.created_at,
        held_back: reasons.get(run.run_id) ?? null,
        point_count: storedPointsOf(run),
        run_id: run.run_id,
        ...runStatus(run, pointCount),
      };
    });
  };

  /** Finishes the running item of the model's oldest unsettled run and starts the next one. */
  const advance = (modelKey) => {
    if (clock() - lastAdvanceAt < STEP_MS) {
      return;
    }
    lastAdvanceAt = clock();

    const items = (runs.get(modelKey) ?? []).flatMap((run) => run.items.map((item) => ({ item, run })));
    const running = items.find(({ item }) => item.status === 'in_progress');

    if (running) {
      running.item.status = 'completed';
      running.run.lastActivityAt = clock();
      addPoint(modelKey, running.run, running.item);
    }

    const next = items.find(({ item }) => item.status === 'pending');

    if (next) {
      next.item.status = 'in_progress';
    }
  };

  /**
   * A model's lines: its own points, plus its partners' points in the current environment, which the server pools
   * the same way. A line only partners measured appears with `own_windows` 0.
   */
  const linesOf = (modelKey, scenario) => {
    const reasons = heldBack();
    // A window of an aborted attempt never counts, whatever else it measured.
    const judged = (point, key) => ({
      ...point,
      excluded: reasons.get(point.run_id) === 'aborted_attempt' ? 'aborted_attempt' : point.excluded,
      model_key: key,
    });
    const own = (points.get(modelKey) ?? []).map((point) => judged(point, modelKey));
    const shared = partnersOf(modelKey).flatMap((key) =>
      (points.get(key) ?? []).filter((point) => point.env_id === ENV_ID).map((point) => judged(point, key))
    );

    return DTYPES.flatMap((dtype) => {
      const line = `decode_${dtype}`;
      const ownPoints = own.filter((point) => point.line === line);
      const envId = ownPoints.at(-1)?.env_id ?? ENV_ID;
      const linePoints = [...ownPoints, ...(envId === ENV_ID ? shared.filter((point) => point.line === line) : [])];

      if (linePoints.length === 0) {
        return [];
      }

      // Only counted windows feed the fit; a size counts while any of its windows does.
      const countedPoints = linePoints.filter((point) => point.excluded === undefined);
      const sides = [...new Set(linePoints.map((point) => point.width))].sort((left, right) => left - right);
      const countedSides = new Set(countedPoints.map((point) => point.width));
      const fitted = countedSides.size >= 2;
      const maxCounted = Math.max(...countedSides);
      const sizes = sides.map((side) => {
        const counted = countedSides.has(side);
        const atSide = (counted ? countedPoints : linePoints).filter((point) => point.width === side);
        const largest = atSide.reduce((left, right) => (right.bytes > left.bytes ? right : left));
        const pixels = side * side;
        const demand = demandOf(pixels, dtype);
        const shipped = Math.round(demand * 1.35);
        const line = Math.round(demand * 1.05);

        const reasonsAtSide = new Set(atSide.map((point) => point.excluded));

        return {
          counted,
          // As the server names a size all of whose windows are left out: aborted where that is the only reason.
          excluded_reason: counted
            ? null
            : reasonsAtSide.size === 1 && reasonsAtSide.has('aborted_attempt')
              ? 'aborted_attempt'
              : reasonsAtSide.has('fragmented')
                ? 'fragmented'
                : 'partial_residency',
          height: side,
          measured_bytes: largest.bytes,
          pixels,
          shipped_bytes: shipped,
          // Like the server's resolve: above the measured range the larger of the user line and the shipped value.
          user_bytes: fitted && envId === ENV_ID ? (side > maxCounted ? Math.max(line, shipped) : line) : null,
          width: side,
          windows: atSide.length,
        };
      });
      const state =
        envId !== ENV_ID ? 'unusable' : !fitted ? 'insufficient' : scenario === 'stale' ? 'stale' : 'in_use';
      const measuredBy = [...new Set(countedPoints.map((point) => point.model_key))]
        .map((key) => ({
          model_key: key,
          name: byKey.get(key)?.name ?? null,
          windows: countedPoints.filter((point) => point.model_key === key).length,
        }))
        .sort((left, right) => left.name.localeCompare(right.name) || left.model_key.localeCompare(right.model_key));

      return [
        {
          base_bytes: fitted ? 420 * MIB : null,
          carrier: 'sd_vae_decode',
          changed_facts: state === 'stale' ? ['driver'] : state === 'unusable' ? ['torch', 'cuda'] : [],
          env_id: envId,
          estimator: 'sd_vae_decode',
          line,
          measured_by: measuredBy,
          measured_range: fitted
            ? { max_pixels: Math.max(...countedSides) ** 2, min_pixels: Math.min(...countedSides) ** 2 }
            : null,
          own_windows: countedPoints.filter((point) => point.model_key === modelKey).length,
          problem: fitted
            ? null
            : countedPoints.length === 0 && linePoints.every((point) => point.excluded === 'aborted_attempt')
              ? ABORTED_LINE_PROBLEM
              : 'A line needs two sizes at least 1.5 times apart.',
          slope_bytes_per_pixel: fitted ? 1_100 : null,
          sizes,
          state,
          version: 1,
        },
      ];
    });
  };

  const unavailableReason = (scenario) =>
    scenario === 'multiGpu'
      ? 'Calibration needs exactly one generation device; this machine generates on cuda:0 and cuda:1.'
      : scenario === 'noGpu'
        ? 'This machine has no GPU to calibrate.'
        : scenario === 'storeUnusable'
          ? `working_memory/${DEVICE_CLASS.id}.json has schema_version 9; this version reads up to 1.`
          : null;

  const plan = (model, scenario) => {
    const base = {
      calibrated_through: null,
      cases: [],
      device_class: null,
      estimated_seconds: null,
      estimators: model.type === 'main' ? ['sd_vae_decode'] : [],
      model_key: model.key,
      not_covered: [],
      reason: null,
      repetitions: REPETITIONS,
      shared_with: [],
      supported: false,
    };

    if (model.type !== 'main') {
      return {
        ...base,
        reason: `Calibration covers the decode of SD1, SD2, SDXL, SD3, FLUX.1, FLUX.2, Z-Image and CogView4 main models and their VAEs; no installed main model generates with this ${model.base} ${model.type}.`,
      };
    }

    if (MODEL_SCENARIOS[model.key] === 'unsupported') {
      return { ...base, reason: 'The inpaint variant does not generate from text alone.' };
    }

    if (scenario === 'multiGpu' || scenario === 'noGpu') {
      return { ...base, reason: unavailableReason(scenario) };
    }

    const cases = casesOf(model);
    const shared = mainModels.filter((other) => other.key !== model.key).slice(0, 3);

    return {
      ...base,
      cases,
      device_class: { id: DEVICE_CLASS.id, name: DEVICE_CLASS.name },
      estimated_seconds: 30 + stepsOf(cases).length * 6,
      not_covered: [ENCODE_NOT_COVERED],
      shared_with: shared.map((other) => ({
        base: other.base,
        key: other.key,
        lines: DTYPES.map((dtype) => `decode_${dtype}`),
        name: other.name,
        type: other.type,
      })),
      supported: true,
    };
  };

  const status = (scenario) => {
    const reason = unavailableReason(scenario);
    const hasClass = scenario !== 'multiGpu' && scenario !== 'noGpu';
    const allRuns = [...runs.entries()].flatMap(([modelKey, list]) =>
      list.map((run) => {
        const pointCount = pointCountOf(run);

        return {
          attempt_id: run.attempt_id,
          attempt_steps: run.attempt_steps,
          created_at: run.created_at,
          env_id: ENV_ID,
          item_ids: run.items.map((item) => item.item_id),
          model_hash: byKey.get(modelKey)?.hash ?? null,
          model_key: modelKey,
          point_count: storedPointsOf(run),
          run_id: run.run_id,
          user_id: 'system',
          ...runStatus(run, pointCount),
        };
      })
    );

    return {
      available: reason === null,
      current_device_class_id: hasClass ? DEVICE_CLASS.id : null,
      current_env_id: hasClass ? ENV_ID : null,
      device_classes: hasClass
        ? [
            {
              device_class: scenario === 'storeUnusable' ? null : DEVICE_CLASS,
              device_class_id: DEVICE_CLASS.id,
              envs:
                scenario === 'storeUnusable'
                  ? []
                  : [
                      {
                        hard: {
                          cuda: '12.8',
                          os: scenario === 'notVerified' ? 'linux' : 'windows',
                          torch: '2.7',
                        },
                        hard_matches_current: true,
                        id: ENV_ID,
                        point_count: [...points.values()].flat().filter((point) => point.env_id === ENV_ID).length,
                        soft: { driver: '576.02' },
                        soft_matches_current: true,
                      },
                    ],
              is_current: true,
              ...measurement(scenario),
              point_counts: [],
              problem: scenario === 'storeUnusable' ? reason : null,
              runs: scenario === 'storeUnusable' ? [] : allRuns,
              usable: scenario !== 'storeUnusable',
            },
          ]
        : [],
      unavailable_reason: reason,
    };
  };

  /** The measurement method of the device class; `notVerified` stands for a backend nobody checked it on. */
  const measurement = (scenario) => ({
    measurement_method: 'allocator_peak',
    measurement_verified: scenario !== 'notVerified',
  });

  const summary = (model, scenario) => {
    advance(model.key);

    return {
      current_env_id: ENV_ID,
      device_class_id: DEVICE_CLASS.id,
      excluded_windows: 0,
      lines: scenario === 'storeUnusable' ? [] : linesOf(model.key, MODEL_SCENARIOS[model.key]),
      ...measurement(scenario),
      model_key: model.key,
      problem: scenario === 'storeUnusable' ? unavailableReason(scenario) : null,
      runs: scenario === 'storeUnusable' ? [] : modelRuns(model.key),
      usable: scenario !== 'storeUnusable',
    };
  };

  const calibrate = (model, body, scenario) => {
    if (unavailableReason(scenario) !== null) {
      return { body: { detail: unavailableReason(scenario) }, status: 409 };
    }

    if (typeof body?.graph?.nodes !== 'object' || body.graph.nodes === null) {
      return { body: { detail: 'The graph cannot be measured as a calibration run.' }, status: 422 };
    }

    const graphModelsRefusal = graphModelsProblem(body.graph, body.graph_models);

    if (graphModelsRefusal !== null) {
      return { body: { detail: graphModelsRefusal }, status: 422 };
    }

    const attemptId = body.attempt_id ?? null;
    const attemptSteps = body.attempt_steps ?? null;

    if (attemptId !== null && !(typeof attemptId === 'string' && ATTEMPT_ID_PATTERN.test(attemptId))) {
      return { body: { detail: 'attempt_id must be 1 to 64 of A-Z a-z 0-9 _ . : -' }, status: 422 };
    }

    if ((attemptId === null) !== (attemptSteps === null)) {
      return { body: { detail: 'attempt_id and attempt_steps are sent together, or neither.' }, status: 422 };
    }

    if (
      attemptSteps !== null &&
      !(Number.isInteger(attemptSteps) && attemptSteps >= 1 && attemptSteps <= MAX_ATTEMPT_STEPS)
    ) {
      return { body: { detail: `attempt_steps must be 1 to ${String(MAX_ATTEMPT_STEPS)}.` }, status: 422 };
    }

    const attemptRuns = (runs.get(model.key) ?? []).filter((run) => attemptId !== null && run.attempt_id === attemptId);

    if (attemptRuns.some((run) => run.attempt_steps !== attemptSteps)) {
      return { body: { detail: "attempt_steps differs from that of the attempt's runs already stored." }, status: 409 };
    }

    const key = typeof body.idempotency_key === 'string' ? body.idempotency_key : null;
    const known = key ? receipts.get(key) : undefined;
    const existing = known && (runs.get(model.key) ?? []).find((run) => run.run_id === known);

    if (!existing) {
      // A later step after another run, of any model or none, was created since the attempt's first: it would measure
      // the GPU after that calibration. The attempt stays short, and so aborted.
      const ownFirst = [...runs.values()]
        .flat()
        .filter((run) => attemptId !== null && run.attempt_id === attemptId)
        .map((run) => run.created_at)
        .sort()[0];

      if (
        ownFirst !== undefined &&
        [...runs.values()].flat().some((run) => run.attempt_id !== attemptId && run.created_at > ownFirst)
      ) {
        return { body: { detail: INTERLEAVED_REFUSAL }, status: 409 };
      }

      // One attempt per device class, as the server judges it: a request with an attempt is refused while another
      // attempt, of any model, runs or is open (`attemptBlocks`). A run without an attempt holds nothing, and a
      // request without one is never refused for it.
      if (attemptId !== null && attemptBlocks(attemptId)) {
        return { body: { detail: RUNNING_REFUSAL }, status: 409 };
      }

      // Every model the graph names must still be the record it compiled over: as its identifier and its
      // graph_models entry name it.
      for (const identifier of modelIdentifiersOf(body.graph)) {
        const record = currentModel ? currentModel(identifier.key) : byKey.get(identifier.key);
        const name = identifier.name ?? record?.name ?? identifier.key;

        if (!record) {
          const message = `${name} was removed after this calibration was planned. Calibrate again.`;
          return { body: { detail: { code: 'model_changed', message } }, status: 409 };
        }

        const compiled = body.graph_models[identifier.key];
        const facts = IDENTITY_FACTS.filter(
          (fact) =>
            (fact in identifier && record[fact] !== identifier[fact]) || (record[fact] ?? null) !== compiled[fact]
        );

        if (facts.length > 0) {
          const message = `${name} changed after this calibration was planned (${facts.join(', ')}). Calibrate again.`;
          return { body: { detail: { code: 'model_changed', message } }, status: 409 };
        }
      }
    }

    const caseId = typeof body.graph.id === 'string' ? body.graph.id : null;
    const step = casesOf(model).find((item) => item.id === caseId) ?? casesOf(model)[0];
    const run = existing ?? createRun(model.key, [step], now(), attemptId, attemptSteps);

    if (key) {
      receipts.set(key, run.run_id);
    }

    return {
      body: { device_class_id: DEVICE_CLASS.id, item_ids: run.items.map((item) => item.item_id), run_id: run.run_id },
      status: 201,
    };
  };

  const cancelByOrigin = (originPrefix) => {
    let canceled = 0;

    for (const list of runs.values()) {
      for (const run of list) {
        if (!`${ORIGIN_PREFIX}${run.run_id}`.startsWith(originPrefix)) {
          continue;
        }

        for (const item of run.items) {
          if (item.status === 'pending' || item.status === 'in_progress') {
            // A decode canceled while it ran leaves an invalid window.
            item.interrupted = item.status === 'in_progress';
            item.status = 'canceled';
            canceled += 1;
          }
        }
      }
    }

    return canceled;
  };

  const resetModel = (modelKey) => {
    const removed = runs.get(modelKey) ?? [];
    const pointsRemoved = (points.get(modelKey) ?? []).length;

    runs.delete(modelKey);
    points.delete(modelKey);

    return { points_removed: pointsRemoved, runs_removed: removed.length };
  };

  /** Answers a working-memory route, or returns null for any other path. */
  const handle = ({ body, method, path, scenario }) => {
    if (path === '/api/v2/models/working_memory/status' && method === 'GET') {
      return scenario === 'error'
        ? { body: { detail: 'The calibration store cannot be read (fixture outage).' }, status: 503 }
        : { body: status(scenario), status: 200 };
    }

    const match = /^\/api\/v2\/models\/i\/([^/]+)\/working_memory(?:\/(plan|summary|calibrate))?$/.exec(path);

    if (!match) {
      return null;
    }

    const model = byKey.get(decodeURIComponent(match[1]));

    if (!model) {
      return { body: { detail: 'The model could not be found' }, status: 404 };
    }

    if (scenario === 'error') {
      return { body: { detail: 'The calibration store cannot be read (fixture outage).' }, status: 503 };
    }

    const noDeviceClass = scenario === 'multiGpu' || scenario === 'noGpu';

    switch (`${method} ${match[2] ?? ''}`) {
      case 'GET plan':
        return { body: plan(model, scenario), status: 200 };
      case 'GET summary':
        return noDeviceClass
          ? { body: { detail: 'No device_class_id was given and there is no current one' }, status: 409 }
          : { body: summary(model, scenario), status: 200 };
      case 'POST calibrate':
        return calibrate(model, body, scenario);
      case 'DELETE ':
        return noDeviceClass
          ? { body: { detail: 'No device_class_id was given and there is no current one' }, status: 409 }
          : { body: resetModel(model.key), status: 200 };
      default:
        return { body: { detail: 'Method Not Allowed' }, status: 405 };
    }
  };

  return {
    cancelByOrigin,
    handle,
    isCalibrationPrefix: (prefix) => prefix.startsWith(ORIGIN_PREFIX),
    reset,
    seedModelChanged,
    seedRunning,
    seedStopped,
  };
};
