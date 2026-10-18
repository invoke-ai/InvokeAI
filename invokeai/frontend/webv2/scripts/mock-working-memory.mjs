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
 * which leaves that size counted on the third. `POST /__faults` with `workingMemory` selects a device scenario instead.
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
/** A run's items advance at most this often, so a section that refetches on every queued step still shows progress. */
const STEP_MS = 1_000;

/** What each seeded main model shows, by key. */
const MODEL_SCENARIOS = {
  'fixture-model-006': 'completed',
  'fixture-model-011': 'stale',
  'fixture-model-016': 'unusable',
  'fixture-model-021': 'running',
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

export const createWorkingMemoryMock = (models, { clock = Date.now } = {}) => {
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

  /** `excluded` names why the fit leaves the window out: `fragmented` or `partial_residency`. */
  const addPoint = (modelKey, item, envId = ENV_ID, excluded = undefined) => {
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
      width: item.case.width,
    });
    points.set(modelKey, list);
  };

  const createRun = (modelKey, steps, createdAt, attemptId) => {
    const run = {
      attempt_id: attemptId,
      created_at: createdAt,
      items: steps.map((step) => ({ case: step, error: null, item_id: nextItemId++, status: 'pending' })),
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
      createRun(model.key, [step], new Date(SEEDED_AT + index * 2_000).toISOString(), `seed-${model.key}`)
    );

    settledRuns.forEach((run, index) => {
      const [item] = run.items;
      const isRunning = scenario === 'running' && index >= 4;

      if (scenario === 'failed' && index === 7) {
        item.status = 'failed';
        item.error = 'OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB';
      } else if (scenario === 'failed' && index > 7) {
        item.status = 'canceled';
      } else if (isRunning) {
        item.status = index === 4 ? 'in_progress' : 'pending';
      } else {
        const repetition = index % (REPETITIONS.warm_up + REPETITIONS.measured);

        item.status = 'completed';
        addPoint(
          model.key,
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

  const runStatus = (run, pointCount) => {
    const statuses = run.items.map((item) => item.status);

    if (statuses.some((status) => status === 'pending' || status === 'in_progress')) {
      return { error: null, status: 'running' };
    }

    const failed = run.items.find((item) => item.status === 'failed');

    if (failed) {
      return { error: failed.error, status: 'failed' };
    }

    if (statuses.every((status) => status === 'completed')) {
      return { error: null, status: 'completed' };
    }

    return { error: null, status: pointCount > 0 ? 'partial' : 'canceled' };
  };

  const pointCountOf = (run) => run.items.filter((item) => item.status === 'completed').length;

  const modelRuns = (modelKey) =>
    (runs.get(modelKey) ?? []).map((run) => {
      const pointCount = pointCountOf(run);

      return {
        attempt_id: run.attempt_id,
        created_at: run.created_at,
        point_count: pointCount,
        run_id: run.run_id,
        ...runStatus(run, pointCount),
      };
    });

  /** Finishes the running item of the model's oldest unsettled run and starts the next one. */
  const advance = (modelKey) => {
    if (clock() - lastAdvanceAt < STEP_MS) {
      return;
    }
    lastAdvanceAt = clock();

    const items = (runs.get(modelKey) ?? []).flatMap((run) => run.items);
    const running = items.find((item) => item.status === 'in_progress');

    if (running) {
      running.status = 'completed';
      addPoint(modelKey, running);
    }

    const next = items.find((item) => item.status === 'pending');

    if (next) {
      next.status = 'in_progress';
    }
  };

  /**
   * A model's lines: its own points, plus its partners' points in the current environment, which the server pools
   * the same way. A line only partners measured appears with `own_windows` 0.
   */
  const linesOf = (modelKey, scenario) => {
    const own = (points.get(modelKey) ?? []).map((point) => ({ ...point, model_key: modelKey }));
    const shared = partnersOf(modelKey).flatMap((key) =>
      (points.get(key) ?? []).filter((point) => point.env_id === ENV_ID).map((point) => ({ ...point, model_key: key }))
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

        return {
          counted,
          excluded_reason: counted ? null : largest.excluded,
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
          problem: fitted ? null : 'A line needs two sizes at least 1.5 times apart.',
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
          created_at: run.created_at,
          env_id: ENV_ID,
          item_ids: run.items.map((item) => item.item_id),
          model_hash: byKey.get(modelKey)?.hash ?? null,
          model_key: modelKey,
          point_count: pointCount,
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

    const attemptId = body.attempt_id ?? null;

    if (attemptId !== null && !(typeof attemptId === 'string' && ATTEMPT_ID_PATTERN.test(attemptId))) {
      return { body: { detail: 'attempt_id must be 1 to 64 of A-Z a-z 0-9 _ . : -' }, status: 422 };
    }

    const key = typeof body.idempotency_key === 'string' ? body.idempotency_key : null;
    const known = key ? receipts.get(key) : undefined;
    const existing = known && (runs.get(model.key) ?? []).find((run) => run.run_id === known);
    const caseId = typeof body.graph.id === 'string' ? body.graph.id : null;
    const step = casesOf(model).find((item) => item.id === caseId) ?? casesOf(model)[0];
    const run = existing ?? createRun(model.key, [step], now(), attemptId);

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

  return { cancelByOrigin, handle, isCalibrationPrefix: (prefix) => prefix.startsWith(ORIGIN_PREFIX), reset };
};
