import type { ModelConfig } from '@features/models/core/types';
import type {
  CalibrationGraphCompiler,
  WorkingMemoryCase,
  WorkingMemoryLine,
  WorkingMemoryPlan,
  WorkingMemoryRun,
  WorkingMemoryStatus,
  WorkingMemorySummary,
} from '@features/models/core/workingMemory';
import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelsSnapshot, refreshModels } from '@features/models/data/modelsStore';
import { WORKING_MEMORY_POLL_MS } from '@features/models/data/workingMemoryQueries';
import { CalibrationGraphCompilerProvider } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const api = vi.hoisted(() => ({
  acknowledgeCalibrationEnqueue: vi.fn((_idempotencyKey: string) => Promise.resolve()),
  calibrateWorkingMemory: vi.fn(),
  cancelCalibrationRun: vi.fn((_runId: string) => Promise.resolve()),
  getWorkingMemoryPlan: vi.fn(),
  getWorkingMemoryStatus: vi.fn(),
  getWorkingMemorySummary: vi.fn(),
  resetWorkingMemory: vi.fn((_modelKey: string, _signal?: AbortSignal) => Promise.resolve()),
}));

vi.mock('@features/models/data/workingMemoryApi', () => api);

const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));

vi.mock('@features/models/ui/useModelsNotify', () => ({ useNotify: () => notify }));

/** The model records the server holds; a compile reads them again first. */
const library = vi.hoisted(() => ({ failNext: null as Error | null, models: [] as unknown[] }));

// Spread over the real module: the models store imports the rest of this transport module.
vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  getModelsDir: () => Promise.resolve(''),
  listMissingModels: () => Promise.resolve([]),
  listModels: () => {
    const failure = library.failNext;
    library.failNext = null;
    return failure ? Promise.reject(failure) : Promise.resolve(library.models);
  },
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      options && Object.keys(options).length > 0
        ? `${key}(${Object.entries(options)
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([name, value]) => `${name}=${String(value)}`)
            .join(',')})`
        : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Loaded with the file, after the mocks above: a first load inside a test spends that test's time budget, which a busy
// machine can exhaust.
const { WorkingMemorySection } = await import('./WorkingMemorySection');

const MODEL_KEY = 'sdxl-main';
const DEVICE = { id: '0123456789abcdef', name: 'Fixture GPU 24GB' };
const T0 = Date.parse('2026-10-01T12:00:00.000Z');

const planCase = (dtype: string, side: number): WorkingMemoryCase => ({
  components: {},
  estimator: 'sdxl_vae_decode',
  height: side,
  id: `sdxl_vae_decode-decode-${dtype}-${String(side)}x${String(side)}`,
  line: `decode_${dtype}`,
  mode: 'txt2img',
  model_key: MODEL_KEY,
  negative_prompt: '',
  positive_prompt: 'a lake',
  seed: 1,
  settings: { vaePrecision: dtype },
  steps: 2,
  width: side,
});

const CASES = [planCase('fp16', 512), planCase('fp32', 512)];

const supportedPlan = (): WorkingMemoryPlan => ({
  calibrated_through: null,
  cases: CASES,
  device_class: DEVICE,
  estimated_seconds: 150,
  estimators: ['sdxl_vae_decode'],
  model_key: MODEL_KEY,
  not_covered: ['encode: not measured'],
  reason: null,
  repetitions: { measured: 2, warm_up: 1 },
  shared_with: [{ base: 'sd-1', key: 'sd1', lines: ['decode_fp16'], name: 'Shared SD1', type: 'main' }],
  supported: true,
});

const status = (overrides: Partial<WorkingMemoryStatus> = {}): WorkingMemoryStatus => ({
  available: true,
  current_device_class_id: DEVICE.id,
  current_env_id: 'env',
  device_classes: [],
  unavailable_reason: null,
  ...overrides,
});

const run = (
  id: string,
  offset: number,
  runStatus: WorkingMemoryRun['status'],
  error: string | null = null,
  attemptId: string | null = 'earlier-attempt',
  overrides: Partial<WorkingMemoryRun> = {}
): WorkingMemoryRun => ({
  attempt_id: attemptId,
  attempt_steps: null,
  created_at: new Date(T0 + offset * 1_000).toISOString(),
  error,
  error_code: null,
  held_back: null,
  // As the server counts: a failed or canceled decode leaves its window too, as an invalid one.
  point_count: runStatus === 'running' ? 0 : 1,
  run_id: id,
  status: runStatus,
  ...overrides,
});

/** How the server marks every run of an attempt it discarded. */
const ABORTED = { held_back: 'aborted_attempt' } as const;

/** The English text of a key, for what the user must read in it. */
const english = (await fetch('/locales/en.json').then((response) => response.json())) as {
  models: { workingMemory: Record<string, string> };
};

const OWN = { model_key: MODEL_KEY, name: 'This SDXL', windows: 6 };

const line = (
  state: WorkingMemoryLine['state'],
  changedFacts: string[] = [],
  overrides: Partial<WorkingMemoryLine> = {}
): WorkingMemoryLine => ({
  base_bytes: 400_000_000,
  carrier: 'sdxl_vae_decode',
  changed_facts: changedFacts,
  env_id: 'env',
  estimator: 'sdxl_vae_decode',
  line: 'decode_fp16',
  measured_by: [OWN],
  measured_range: { max_pixels: 1_048_576, min_pixels: 262_144 },
  problem: null,
  sizes: [
    {
      counted: true,
      excluded_reason: null,
      height: 512,
      measured_bytes: 1_234_000_000,
      pixels: 262_144,
      shipped_bytes: null,
      user_bytes: null,
      width: 512,
      windows: 3,
    },
    {
      counted: true,
      excluded_reason: null,
      height: 1024,
      measured_bytes: 2_100_000_000,
      pixels: 1_048_576,
      shipped_bytes: 2_800_000_000,
      user_bytes: 2_600_000_000,
      width: 1024,
      windows: 3,
    },
  ],
  own_windows: 6,
  slope_bytes_per_pixel: 1_100,
  state,
  version: 1,
  ...overrides,
});

const summary = (overrides: Partial<WorkingMemorySummary> = {}): WorkingMemorySummary => ({
  current_env_id: 'env',
  device_class_id: DEVICE.id,
  excluded_windows: 0,
  lines: [],
  measurement_method: 'allocator',
  measurement_verified: true,
  model_key: MODEL_KEY,
  problem: null,
  runs: [],
  usable: true,
  ...overrides,
});

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;
let compile: ReturnType<typeof vi.fn<CalibrationGraphCompiler>>;
let currentSummary: WorkingMemorySummary | Error;

/** Renders the section, opened as a user would unless `expanded` is false: it starts collapsed. */
const renderSection = async ({ expanded = true }: { expanded?: boolean } = {}) => {
  host = document.createElement('div');
  host.style.width = '720px';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
          <CalibrationGraphCompilerProvider compile={compile}>
            <WorkingMemorySection modelKey={MODEL_KEY} />
          </CalibrationGraphCompilerProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

  if (expanded) {
    await act(() => userEvent.click(sectionToggle()));
  }
};

/** The disclosure button inside the section heading. */
const sectionToggle = () => host.querySelector<HTMLButtonElement>('h3 button')!;

const buttonNamed = (name: string, within: ParentNode = document): HTMLButtonElement | undefined =>
  [...within.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent.trim() === name);

const openDialog = () => document.querySelector<HTMLElement>('[role="dialog"][data-state="open"]');
const openAlertDialog = () => document.querySelector<HTMLElement>('[role="alertdialog"][data-state="open"]');
const sectionText = () => host.textContent;

/** The summary the section reads next; a poll or an invalidation picks it up. */
const serveSummary = (next: WorkingMemorySummary | Error) => {
  currentSummary = next;
};

const refreshSummary = () => act(() => queryClient.invalidateQueries());

/** The calibrated main model's record as the server holds it. */
const mainModel = (base: string): ModelConfig =>
  ({
    base,
    description: null,
    file_size: 1024,
    format: 'checkpoint',
    hash: 'hash',
    key: MODEL_KEY,
    name: 'This SDXL',
    source: MODEL_KEY,
    source_type: 'path',
    type: 'main',
  }) as ModelConfig;

beforeEach(() => {
  document.documentElement.lang = 'en';
  accountLifecycle.activate('wm-section-test', ':user:wm-section-test');
  library.models = [mainModel('sdxl')];
  // Like Generate's compiler, it builds over the records it is given, rejecting with their read's error; each graph
  // names the base it was built for, and its model in a loader as the record has it.
  compile = vi.fn<CalibrationGraphCompiler>(async (cases, models) => {
    const records = await models;

    return cases.map((item) => {
      const model = records.find((record) => record.key === item.model_key);
      const loader = model
        ? {
            id: 'loader',
            model: { base: model.base, hash: model.hash, key: model.key, name: model.name, type: model.type },
          }
        : null;

      return {
        caseId: item.id,
        data: [],
        graph: { base: model?.base ?? null, edges: [], id: item.id, nodes: loader ? { loader } : {} },
      };
    });
  });
  currentSummary = summary();
  api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(supportedPlan()));
  api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(status()));
  api.getWorkingMemorySummary.mockImplementation(() =>
    currentSummary instanceof Error ? Promise.reject(currentSummary) : Promise.resolve(currentSummary)
  );
  let runNumber = 0;
  api.calibrateWorkingMemory.mockImplementation(() => {
    runNumber += 1;
    return Promise.resolve({ device_class_id: DEVICE.id, item_ids: [runNumber], run_id: `run-${String(runNumber)}` });
  });
});

afterEach(async () => {
  vi.useRealTimers();
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
  vi.clearAllMocks();
});

describe('WorkingMemorySection', () => {
  it('calibrates a model that was never calibrated: dialog, queued runs, progress, then the measurements', async () => {
    // The summary polls on an interval while a run is running: the test steps that interval rather than wait it out.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');

    const section = host.querySelector('section')!;
    expect(section.getAttribute('aria-labelledby')).toBe(host.querySelector('h3')!.id);
    expect(host.querySelector('h3')!.textContent).toBe('models.workingMemory.title');
    expect(sectionText()).toContain(DEVICE.name);
    expect(sectionText()).toContain('models.workingMemory.stateNone');
    // The compiler loads only once Calibrate is clicked.
    expect(compile).not.toHaveBeenCalled();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    // The dialog opens only on a plan that compiles.
    expect(compile).toHaveBeenCalledTimes(1);
    const dialog = openDialog()!;
    expect(dialog.textContent).toContain('models.workingMemory.dialogDuration(minutes=3)');
    expect(dialog.textContent).toContain('models.workingMemory.dialogQueueNote');
    expect(dialog.textContent).toContain('models.workingMemory.dialogSizesNote');
    expect(dialog.textContent).toContain(
      'models.workingMemory.pair(detail=fp16,label=models.workingMemory.size(height=512,width=512))'
    );
    expect(dialog.textContent).toContain(
      'models.workingMemory.pair(detail=fp32,label=models.workingMemory.size(height=512,width=512))'
    );
    expect(dialog.textContent).toContain('models.workingMemory.dialogRuns(count=6)');
    expect(dialog.textContent).toContain('Shared SD1');

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);

    // Compiled to open the dialog, and again on confirmation over the model records read then.
    expect(compile).toHaveBeenCalledTimes(2);
    expect(compile).toHaveBeenLastCalledWith(CASES, expect.any(Promise), expect.any(AbortSignal));
    const requests = api.calibrateWorkingMemory.mock.calls.map(
      ([key, request]) => [key, (request as { graph: { id: string } }).graph.id] as const
    );
    // Each precision warms up first: fp16 warm-up, two measured, then the same in fp32.
    expect(requests).toEqual([
      [MODEL_KEY, CASES[0]!.id],
      [MODEL_KEY, CASES[0]!.id],
      [MODEL_KEY, CASES[0]!.id],
      [MODEL_KEY, CASES[1]!.id],
      [MODEL_KEY, CASES[1]!.id],
      [MODEL_KEY, CASES[1]!.id],
    ]);
    const keys = api.calibrateWorkingMemory.mock.calls.map(
      ([, request]) => (request as { idempotencyKey: string }).idempotencyKey
    );
    expect(new Set(keys).size).toBe(6);
    // One attempt id for the click, on every step and inside every retry key.
    const attemptIds = new Set(
      api.calibrateWorkingMemory.mock.calls.map(([, request]) => (request as { attemptId: string }).attemptId)
    );
    expect(attemptIds.size).toBe(1);
    const [attemptId] = [...attemptIds];
    expect(keys.every((key) => key.includes(attemptId!))).toBe(true);
    expect(api.acknowledgeCalibrationEnqueue.mock.calls.map(([key]) => key)).toEqual(keys);

    serveSummary(
      summary({
        runs: [
          run('run-1', 0, 'completed', null, attemptId),
          run('run-2', 1, 'completed', null, attemptId),
          ...[3, 4, 5, 6].map((number) => run(`run-${String(number)}`, number, 'running', null, attemptId)),
        ],
      })
    );
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.running(done=2,total=6)');
    expect(host.querySelector('[role="progressbar"]')?.getAttribute('aria-label')).toBe(
      'models.workingMemory.progressLabel'
    );
    expect(sectionText()).toContain(
      'models.workingMemory.pair(detail=fp16,label=models.workingMemory.size(height=512,width=512))'
    );
    expect(buttonNamed('models.workingMemory.recalibrate')?.disabled).toBe(true);
    expect(sectionText()).toContain('models.workingMemory.blockedRunning');
    // Reset names the same reason it is disabled for.
    const reset = buttonNamed('common.reset')!;
    expect(reset.disabled).toBe(true);
    expect(document.getElementById(reset.getAttribute('aria-describedby')!)?.textContent).toBe(
      'models.workingMemory.blockedRunning'
    );

    serveSummary(
      summary({
        lines: [line('in_use')],
        // An earlier attempt failed; the newest attempt is the one that counts.
        runs: [
          run('old', -600, 'failed', 'old failure'),
          ...[1, 2, 3, 4, 5, 6].map((number) => run(`run-${String(number)}`, number, 'completed', null, attemptId)),
        ],
      })
    );
    // Polling picks the end up without any other trigger, one poll interval later.
    await act(() => vi.advanceTimersByTimeAsync(WORKING_MEMORY_POLL_MS));
    await expect.poll(() => host.querySelector('table')).not.toBeNull();

    const table = host.querySelector('table')!;
    expect(table.getAttribute('aria-label')).toBe(
      'models.workingMemory.tableLabel(line=models.workingMemory.pair(detail=fp16,label=models.workingMemory.operationDecode))'
    );
    const rows = [...table.querySelectorAll('tbody tr')].map((row) =>
      [...row.querySelectorAll('th, td')].map((cell) => cell.textContent)
    );
    expect([...table.querySelectorAll('tbody th')].map((cell) => cell.getAttribute('scope'))).toEqual(['row', 'row']);
    expect(table.querySelectorAll('thead th')[3]?.textContent).toBe('models.workingMemory.columnCalibrated');
    // The size labels its row on the left; every value sits right-aligned under its header.
    const alignments = (selector: string) =>
      [...table.querySelectorAll(selector)].map((cell) => getComputedStyle(cell).textAlign);
    expect(alignments('thead th')).toEqual(['start', 'end', 'end', 'end']);
    expect(alignments('tbody tr:first-child > *')).toEqual(['start', 'end', 'end', 'end']);
    // Missing estimates show a dash; sizes format like the rest of the app.
    expect(rows).toEqual([
      ['models.workingMemory.size(height=512,width=512)', '1.2 GB', '—', '—'],
      ['models.workingMemory.size(height=1024,width=1024)', '2.1 GB', '2.8 GB', '2.6 GB'],
    ]);
    expect(sectionText()).toContain('models.workingMemory.measuredRange(max=1.05,min=0.26)');
    expect(sectionText()).toContain('models.workingMemory.stateInUse');
    expect(host.querySelector('[role="progressbar"]')).toBeNull();
  });

  it('operates the dialog by keyboard and returns focus to its opener', async () => {
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();

    const opener = buttonNamed('models.workingMemory.calibrate')!;
    opener.focus();
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.poll(openDialog).not.toBeNull();
    await expect
      .poll(() => openDialog()?.contains(document.activeElement) && document.activeElement?.textContent)
      .toBe('models.workingMemory.calibrate');
    expect(openDialog()!.querySelector('h2, [id$="title"]')?.textContent).toBe('models.workingMemory.dialogTitle');

    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(openDialog).toBeNull();
    await expect.poll(() => document.activeElement).toBe(opener);
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{Enter}'));
    await expect
      .poll(() => openDialog()?.contains(document.activeElement) && document.activeElement?.textContent)
      .toBe('models.workingMemory.calibrate');
    // Hold the first step's response so the attempt stays in its running state while focus settles.
    let releasePost = () => undefined as void;
    const held = new Promise<void>((resolve) => {
      releasePost = resolve;
    });
    const post = api.calibrateWorkingMemory.getMockImplementation()!;
    api.calibrateWorkingMemory.mockImplementationOnce(async (...args: unknown[]) => {
      await held;
      return post(...args);
    });
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.poll(openDialog).toBeNull();
    // Starting disables the opener; focus moves to the action that is available now instead of the page.
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.stop');
    expect(host.contains(document.activeElement)).toBe(true);

    await act(() => releasePost());
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);
  });

  it('stops a running calibration after confirmation, newest run first', async () => {
    serveSummary(
      summary({ runs: [run('run-1', 0, 'completed'), run('run-2', 1, 'running'), run('run-3', 2, 'running')] })
    );
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.stop')).toBeDefined();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop')!));
    await expect.poll(openAlertDialog).not.toBeNull();
    // The warning describes the dialog, so a screen reader reads it as the dialog opens.
    const describedBy = openAlertDialog()!.getAttribute('aria-describedby') ?? '';
    expect(document.getElementById(describedBy)?.textContent).toBe('models.workingMemory.stopBody');
    // Stop discards the whole attempt, its finished runs too; the user reads that before confirming.
    expect(english.models.workingMemory.stopBody).toMatch(/measurements are discarded, finished runs included/);
    expect(api.cancelCalibrationRun).not.toHaveBeenCalled();

    api.cancelCalibrationRun.mockImplementation(() => {
      serveSummary(
        summary({
          runs: [
            run('run-1', 0, 'completed', null, 'earlier-attempt', ABORTED),
            run('run-2', 1, 'partial', null, 'earlier-attempt', ABORTED),
            run('run-3', 2, 'canceled', null, 'earlier-attempt', ABORTED),
          ],
        })
      );
      return Promise.resolve();
    });
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop', openAlertDialog()!)!));
    await expect.poll(openAlertDialog).toBeNull();

    expect(api.cancelCalibrationRun.mock.calls.map(([runId]) => runId)).toEqual(['run-3', 'run-2']);
    await expect.poll(sectionText).toContain('models.workingMemory.partial');
    expect(sectionText()).toContain('models.workingMemory.discarded');
    // The Stop button is gone; focus lands on the action that is available now.
    await expect.poll(() => document.activeElement?.textContent).toBe('common.retry');
  });

  it('reports an attempt stopped while queuing as canceled, not as queued short', async () => {
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();

    // The third step's response arrives only once the stop is under way.
    let releasePost = () => undefined as void;
    const held = new Promise<void>((resolve) => {
      releasePost = resolve;
    });
    const post = api.calibrateWorkingMemory.getMockImplementation()!;
    api.calibrateWorkingMemory.mockImplementation(async (...args: unknown[]) => {
      if (api.calibrateWorkingMemory.mock.calls.length === 3) {
        await held;
      }
      return post(...args);
    });
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(3);

    const attemptId = (api.calibrateWorkingMemory.mock.calls[0]![1] as { attemptId: string }).attemptId;
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop')!));
    await expect.poll(openAlertDialog).not.toBeNull();
    serveSummary(
      summary({
        runs: [1, 2, 3].map((number) =>
          run(`run-${String(number)}`, number, 'completed', null, attemptId, { ...ABORTED, attempt_steps: 6 })
        ),
      })
    );
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop', openAlertDialog()!)!));
    await act(() => releasePost());
    await expect.poll(openAlertDialog).toBeNull();

    expect(api.calibrateWorkingMemory).toHaveBeenCalledTimes(3);
    await expect.poll(sectionText).toContain('models.workingMemory.partialCanceled');
    expect(sectionText()).not.toContain('models.workingMemory.partialQueued');
    // Stopped after three of six steps: the server holds the attempt back, and its three finished steps are lost.
    expect(sectionText()).toContain('models.workingMemory.discarded');
  });

  it('says the measurements of an attempt the server discarded are gone, offering to calibrate again', async () => {
    // After a reload: a queue clear canceled the attempt's later runs, so the server holds all of them back.
    const planned = { ...ABORTED, attempt_steps: 6 };
    serveSummary(
      summary({
        runs: [
          run('run-1', 0, 'completed', null, 'attempt', planned),
          run('run-2', 1, 'completed', null, 'attempt', planned),
          ...[3, 4, 5, 6].map((number) => run(`run-${String(number)}`, number, 'canceled', null, 'attempt', planned)),
        ],
      })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.partial');

    expect(sectionText()).toContain('models.workingMemory.partialCanceled');
    expect(sectionText()).toContain('models.workingMemory.discarded');
    expect(english.models.workingMemory.discarded).toMatch(/discarded/);
    expect(buttonNamed('common.retry')?.disabled).toBe(false);

    // A step that failed out of memory ends its run, not the attempt: the other steps' measurements still count.
    const ranOn = { attempt_steps: 6 };
    serveSummary(
      summary({
        runs: [
          run('run-1', 0, 'completed', null, 'attempt', ranOn),
          run('run-2', 1, 'failed', 'CUDA out of memory', 'attempt', ranOn),
          ...[3, 4, 5, 6].map((number) => run(`run-${String(number)}`, number, 'completed', null, 'attempt', ranOn)),
        ],
      })
    );
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.failed');
    expect(sectionText()).not.toContain('models.workingMemory.discarded');
    expect(sectionText()).toContain('models.workingMemory.failedKept');
  });

  it('claims no loss for an attempt canceled before it measured anything', async () => {
    // Stop canceled the first step during its decode, which left an invalid window, and the rest while they waited:
    // the server holds the attempt back, but nothing usable was measured.
    const planned = { ...ABORTED, attempt_steps: 6 };
    serveSummary(
      summary({
        runs: [
          run('run-1', 1, 'canceled', null, 'attempt', planned),
          ...[2, 3, 4, 5, 6].map((number) =>
            run(`run-${String(number)}`, number, 'canceled', null, 'attempt', { ...planned, point_count: 0 })
          ),
        ],
      })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.canceled');

    expect(sectionText()).not.toContain('models.workingMemory.discarded');
  });

  it('says nothing still counts when every step failed', async () => {
    // Every decode ran out of memory; each left an invalid window.
    const planned = { attempt_steps: 6 };
    serveSummary(
      summary({
        runs: [1, 2, 3, 4, 5, 6].map((number) =>
          run(`run-${String(number)}`, number, 'failed', 'CUDA out of memory', 'attempt', planned)
        ),
      })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.failed');

    expect(sectionText()).toContain('CUDA out of memory');
    expect(sectionText()).not.toContain('models.workingMemory.failedKept');
    expect(sectionText()).not.toContain('models.workingMemory.discarded');
  });

  it('says a submission that failed partway discarded the measurements of the runs it queued', async () => {
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();
    expect(host.querySelector('[role="status"]')?.textContent).toBe('');
    const post = api.calibrateWorkingMemory.getMockImplementation()!;
    api.calibrateWorkingMemory.mockImplementation((...args: unknown[]) =>
      api.calibrateWorkingMemory.mock.calls.length === 3
        ? Promise.reject(new ApiError('{"detail":"The queue has no room for the run"}', 409))
        : post(...args)
    );

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(3);

    // The server lists the two runs queued before the refusal.
    const attemptId = (api.calibrateWorkingMemory.mock.calls[0]![1] as { attemptId: string }).attemptId;
    serveSummary(
      summary({
        runs: [1, 2].map((number) =>
          run(`run-${String(number)}`, number, 'completed', null, attemptId, { ...ABORTED, attempt_steps: 6 })
        ),
      })
    );
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.partial');
    expect(sectionText()).toContain('The queue has no room for the run');
    expect(sectionText()).toContain('models.workingMemory.discarded');
    // The outcome appears inside a live region that was there before, so it is announced.
    expect(host.querySelector('[role="status"]')?.textContent).toContain('The queue has no room for the run');
    expect(buttonNamed('common.retry')?.disabled).toBe(false);
  });

  it('heads the section with the current environment’s state, not an old environment’s', async () => {
    serveSummary(
      summary({
        lines: [line('unusable', ['torch'], { env_id: 'old-env' }), line('in_use')],
        runs: [run('run-1', 0, 'completed')],
      })
    );
    await renderSection();
    await expect.poll(() => host.querySelectorAll('table').length).toBe(2);

    const header = [...host.querySelectorAll('p')].find(
      (element) => element.textContent === DEVICE.name
    )!.parentElement!;
    expect(header.textContent).toBe(`${DEVICE.name}models.workingMemory.stateInUse`);
    expect(sectionText()).not.toContain('models.workingMemory.neverCalibrated');

    // With only an old environment's lines, they head it, and this GPU counts as not calibrated.
    serveSummary(summary({ lines: [line('unusable', ['torch'], { env_id: 'old-env' })] }));
    await refreshSummary();
    await expect.poll(() => header.textContent).toBe(`${DEVICE.name}models.workingMemory.stateUnusable`);
    expect(sectionText()).toContain('models.workingMemory.neverCalibrated');
  });

  it('shows lines other models measured as in use, naming them, with nothing of this model to reset', async () => {
    const other = { model_key: 'other', name: 'Other SDXL', windows: 6 };
    serveSummary(summary({ lines: [line('in_use', [], { measured_by: [other], own_windows: 0 })] }));
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.usesOthers');

    expect(sectionText()).toContain('models.workingMemory.measuredBy(models=Other SDXL)');
    expect(sectionText()).not.toContain('models.workingMemory.neverCalibrated');
    expect(buttonNamed('common.reset')).toBeUndefined();
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);

    // Once this model measured too, it is listed first and has measurements to reset.
    serveSummary(
      summary({
        lines: [line('in_use', [], { measured_by: [other, OWN], own_windows: 6 })],
        runs: [run('run-1', 0, 'completed')],
      })
    );
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.measuredBy(models=This SDXL, Other SDXL)');
    expect(sectionText()).not.toContain('models.workingMemory.usesOthers');
    expect(buttonNamed('common.reset')).toBeDefined();
  });

  it('keeps a finished attempt completed through a lost connection to the server', async () => {
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);

    const attemptId = (api.calibrateWorkingMemory.mock.calls[0]![1] as { attemptId: string }).attemptId;
    const finished = summary({
      lines: [line('in_use')],
      runs: [1, 2, 3, 4, 5, 6].map((number) => run(`run-${String(number)}`, number, 'completed', null, attemptId)),
    });
    serveSummary(finished);
    await refreshSummary();
    await expect.poll(() => buttonNamed('models.workingMemory.recalibrate')).toBeDefined();

    // The server goes away: every read fails as the browser reports an unreachable host.
    const unreachable = () => Promise.reject(new TypeError('Failed to fetch'));
    api.getWorkingMemoryPlan.mockImplementation(unreachable);
    api.getWorkingMemoryStatus.mockImplementation(unreachable);
    serveSummary(new TypeError('Failed to fetch'));
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');

    // It comes back with the same runs.
    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(supportedPlan()));
    api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(status()));
    serveSummary(finished);
    await refreshSummary();
    await expect.poll(() => buttonNamed('models.workingMemory.recalibrate')).toBeDefined();
    expect(sectionText()).not.toContain('models.workingMemory.failed');
    expect(sectionText()).not.toContain('Failed to fetch');
  });

  it('lets an attempt the server refused before it queued anything give way to one that ran after it', async () => {
    const earlier = run('old', 0, 'completed', null, 'old');
    serveSummary(summary({ lines: [line('in_use')], runs: [earlier] }));
    api.calibrateWorkingMemory.mockImplementationOnce(() =>
      Promise.reject(new ApiError('{"detail":"The queue has no room for the run"}', 409))
    );
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.recalibrate')).toBeDefined();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.recalibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(sectionText).toContain('models.workingMemory.failed');
    expect(sectionText()).toContain('The queue has no room for the run');

    // Another tab, or a script, then calibrated to the end: that attempt is the latest one now.
    serveSummary(
      summary({
        lines: [line('in_use')],
        runs: [
          earlier,
          ...[1, 2, 3, 4, 5, 6].map((number) => run(`theirs-${String(number)}`, 100 + number, 'completed')),
        ],
      })
    );
    await refreshSummary();
    await expect.poll(sectionText).not.toContain('models.workingMemory.failed');
    expect(sectionText()).not.toContain('The queue has no room for the run');
    expect(buttonNamed('models.workingMemory.recalibrate')?.disabled).toBe(false);
  });

  it('keeps a size the fit leaves out in the table, muted, naming why, in columns the other tables share', async () => {
    const counted = line('in_use').sizes;
    serveSummary(
      summary({
        lines: [
          // Its row headers are narrower than the other table's, as a size note makes them in any language.
          line('in_use', [], { sizes: counted.map((size) => ({ ...size, height: 64, width: 64 })) }),
          line('in_use', [], {
            line: 'decode_fp32',
            sizes: [
              ...counted,
              {
                counted: false,
                excluded_reason: 'fragmented',
                height: 1216,
                measured_bytes: 5_900_000_000,
                pixels: 1_478_656,
                shipped_bytes: 4_100_000_000,
                user_bytes: 3_300_000_000,
                width: 1216,
                windows: 3,
              },
            ],
          }),
          // Only an aborted attempt measured this line: none of its sizes counts, and the server says why.
          line('insufficient', [], {
            line: 'decode_bf16',
            problem: 'Stop or an interrupted submission discarded the measurements of this line.',
            sizes: counted.map((size) => ({ ...size, counted: false, excluded_reason: 'aborted_attempt' as const })),
          }),
        ],
        runs: [run('run-1', 0, 'completed')],
      })
    );
    await renderSection();
    await expect.poll(() => host.querySelectorAll('table').length).toBe(3);

    const [plain, mixed, discarded] = [...host.querySelectorAll('table')];
    const [first, , excluded] = [...mixed!.querySelectorAll('tbody tr')];
    expect([...discarded!.querySelectorAll('tbody th')].map((cell) => cell.textContent)).toEqual([
      'models.workingMemory.size(height=512,width=512)models.workingMemory.notCountedAborted',
      'models.workingMemory.size(height=1024,width=1024)models.workingMemory.notCountedAborted',
    ]);
    const stoppedLine = discarded!.closest('table')!.parentElement!.parentElement!;
    expect(stoppedLine.textContent).toContain('models.workingMemory.stateInterrupted');
    expect(stoppedLine.textContent).toContain('models.workingMemory.helpInterrupted');
    expect(stoppedLine.textContent).not.toContain('models.workingMemory.stateInsufficient');
    expect(sectionText()).not.toContain('Stop or an interrupted submission discarded the measurements of this line.');
    // The reason is part of the row header, so it is read with the size.
    expect(excluded!.querySelector('th')!.textContent).toBe(
      'models.workingMemory.size(height=1216,width=1216)models.workingMemory.notCountedFragmented'
    );
    expect(first!.querySelector('th')!.textContent).toBe('models.workingMemory.size(height=512,width=512)');
    const colors = (row: Element) => [...row.children].map((cell) => getComputedStyle(cell).color);
    expect(new Set(colors(excluded!)).size).toBe(1);
    expect(colors(excluded!)[0]).not.toBe(colors(first!)[0]);
    expect([...excluded!.children].map((cell) => getComputedStyle(cell).textAlign)).toEqual([
      'start',
      'end',
      'end',
      'end',
    ]);
    // The note under the size does not widen its column: both tables' columns start at the same place.
    const columnEdges = (table: Element) =>
      [...table.querySelectorAll('thead th')].map((cell) => Math.round(cell.getBoundingClientRect().left));
    expect(columnEdges(mixed!)).toEqual(columnEdges(plain!));
  });

  it('explains each value column from an info button that opens by keyboard', async () => {
    serveSummary(summary({ lines: [line('in_use')], runs: [run('run-1', 0, 'completed')] }));
    await renderSection();
    await expect.poll(() => host.querySelector('table')).not.toBeNull();

    const headers = [...host.querySelectorAll('thead th')];
    // The size column needs no explanation.
    expect(headers[0]!.querySelector('button')).toBeNull();

    for (const [index, column, hint] of [
      [1, 'columnMeasured', 'hintMeasured'],
      [2, 'columnShipped', 'hintShipped'],
      [3, 'columnCalibrated', 'hintCalibrated'],
    ] as const) {
      const button = headers[index]!.querySelector('button')!;
      const name = `models.workingMemory.aboutColumn(column=models.workingMemory.${column})`;
      expect(button.getAttribute('aria-label')).toBe(name);

      button.focus();
      await act(() => userEvent.keyboard('{Enter}'));
      await expect.poll(openDialog).not.toBeNull();
      expect(openDialog()!.getAttribute('aria-label')).toBe(name);
      expect(openDialog()!.textContent).toBe(`models.workingMemory.${hint}`);

      await act(() => userEvent.keyboard('{Escape}'));
      await expect.poll(openDialog).toBeNull();
      await expect.poll(() => document.activeElement).toBe(button);
    }
  });

  it('starts collapsed, keeps its panel out of the tab order, and asks for the plan only once opened', async () => {
    serveSummary(summary({ lines: [line('in_use')], runs: [run('run-1', 0, 'completed')] }));
    await renderSection({ expanded: false });

    const toggle = sectionToggle();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    const panel = document.getElementById(toggle.getAttribute('aria-controls')!)!;
    expect(panel.hidden).toBe(true);
    expect(buttonNamed('models.workingMemory.recalibrate')).toBeUndefined();
    // The plan reads the GPU's free memory on the server; a collapsed section never asks for it.
    await expect.poll(() => api.getWorkingMemorySummary.mock.calls.length).toBeGreaterThan(0);
    expect(api.getWorkingMemoryPlan).not.toHaveBeenCalled();

    toggle.focus();
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.poll(() => buttonNamed('models.workingMemory.recalibrate')).toBeDefined();
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(panel.hidden).toBe(false);
    expect(api.getWorkingMemoryPlan).toHaveBeenCalledTimes(1);

    await act(() => userEvent.keyboard('[Space]'));
    await expect.poll(() => panel.hidden).toBe(true);
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect([...panel.querySelectorAll('button')].every((button) => button.getClientRects().length === 0)).toBe(true);
    await act(() => userEvent.tab());
    expect(panel.contains(document.activeElement)).toBe(false);
  });

  it('names the GPU and the state in the collapsed header, and the progress while a calibration runs', async () => {
    api.getWorkingMemoryStatus.mockImplementation(() =>
      Promise.resolve(
        status({
          device_classes: [
            {
              device_class: {
                arch: null,
                backend: 'cuda',
                id: DEVICE.id,
                name: DEVICE.name,
                total_mib: 24_576,
                vendor: 'x',
              },
              device_class_id: DEVICE.id,
              envs: [],
              is_current: true,
              measurement_method: 'allocator',
              measurement_verified: true,
              point_counts: [],
              problem: null,
              runs: [],
              usable: true,
            },
          ],
        })
      )
    );
    serveSummary(summary({ lines: [line('in_use')], runs: [run('run-1', 0, 'completed')] }));
    await renderSection({ expanded: false });

    const header = sectionToggle().closest('h3')!.parentElement!;
    await expect
      .poll(() => header.textContent)
      .toBe(`models.workingMemory.title${DEVICE.name}models.workingMemory.stateInUse`);

    // The server stores the attempt's step count with its runs, so the header counts the steps not yet queued too
    // without a plan.
    const planned = { attempt_steps: 6 };
    serveSummary(
      summary({
        runs: [
          run('run-1', 1, 'completed', null, 'attempt', planned),
          run('run-2', 2, 'completed', null, 'attempt', planned),
          run('run-3', 3, 'running', null, 'attempt', planned),
          run('run-4', 4, 'running', null, 'attempt', planned),
        ],
      })
    );
    await refreshSummary();
    await expect.poll(() => header.textContent).toContain('models.workingMemory.running(done=2,total=6)');
    expect(header.textContent).not.toContain('models.workingMemory.stateInUse');
    // A running calibration does not open the section.
    expect(sectionToggle().getAttribute('aria-expanded')).toBe('false');
    expect(api.getWorkingMemoryPlan).not.toHaveBeenCalled();
  });

  it('says the model was removed when the server no longer knows it', async () => {
    serveSummary(new ApiError('{"detail":"The model could not be found"}', 404));
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.modelRemoved');

    expect(sectionText()).not.toContain('models.workingMemory.loadFailed');
    expect(buttonNamed('common.retry')).toBeUndefined();
  });

  it('resets after confirmation and restores focus', async () => {
    serveSummary(summary({ lines: [line('in_use')], runs: [run('run-1', 0, 'completed')] }));
    await renderSection();
    await expect.poll(() => buttonNamed('common.reset')).toBeDefined();

    await act(() => userEvent.click(buttonNamed('common.reset')!));
    await expect.poll(openAlertDialog).not.toBeNull();
    // A destructive confirmation opens on Cancel.
    expect(document.activeElement?.textContent).toBe('common.cancel');

    api.resetWorkingMemory.mockImplementation(() => {
      serveSummary(summary());
      return Promise.resolve();
    });
    await act(() => userEvent.click(buttonNamed('common.reset', openAlertDialog()!)!));
    await expect.poll(openAlertDialog).toBeNull();

    expect(api.resetWorkingMemory).toHaveBeenCalledWith(MODEL_KEY, expect.anything());
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.calibrate');
  });

  it('disables Calibrate for an unsupported model, naming the reason', async () => {
    api.getWorkingMemoryPlan.mockImplementation(() =>
      Promise.resolve({
        ...supportedPlan(),
        cases: [],
        reason: 'The inpaint variant does not generate from text alone.',
        supported: false,
      })
    );
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();

    const button = buttonNamed('models.workingMemory.calibrate')!;
    expect(button.disabled).toBe(true);
    expect(document.getElementById(button.getAttribute('aria-describedby')!)?.textContent).toBe(
      'The inpaint variant does not generate from text alone.'
    );
    expect(sectionText()).not.toContain('models.workingMemory.neverCalibrated');
  });

  it('disables Calibrate without exactly one GPU, naming the status reason', async () => {
    api.getWorkingMemoryStatus.mockImplementation(() =>
      Promise.resolve(
        status({
          available: false,
          current_device_class_id: null,
          unavailable_reason: 'Calibration needs exactly one generation device; this machine has two.',
        })
      )
    );
    serveSummary(new ApiError('{"detail":"No device_class_id was given and there is no current one"}', 409));
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();

    const button = buttonNamed('models.workingMemory.calibrate')!;
    expect(button.disabled).toBe(true);
    expect(document.getElementById(button.getAttribute('aria-describedby')!)?.textContent).toBe(
      'Calibration needs exactly one generation device; this machine has two.'
    );
    // A missing device class is a state, not a load failure.
    expect(sectionText()).not.toContain('models.workingMemory.loadFailed');
  });

  it('names an unusable store and flags hardware the method was not verified on', async () => {
    serveSummary(
      summary({
        measurement_verified: false,
        problem: 'schema_version 9 is newer than this version reads',
        usable: false,
      })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.notVerified');

    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(true);
    expect(sectionText()).toContain(
      'models.workingMemory.blockedStoreUnusable(problem=schema_version 9 is newer than this version reads)'
    );
  });

  it('explains stale measurements and offers to recalibrate', async () => {
    serveSummary(
      summary({ lines: [line('stale', ['driver', 'setting.attention_type'])], runs: [run('run-1', 0, 'completed')] })
    );
    await renderSection();
    await expect
      .poll(sectionText)
      .toContain('models.workingMemory.helpStale(facts=models.workingMemory.facts.driver, attention_type)');

    expect(sectionText()).toContain('models.workingMemory.stateStale');
    expect(buttonNamed('models.workingMemory.recalibrate')?.disabled).toBe(false);
  });

  it('reports a failed attempt with its error and offers Retry', async () => {
    serveSummary(summary({ runs: [run('run-1', 0, 'completed'), run('run-2', 1, 'failed', 'CUDA out of memory')] }));
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.failed');

    expect(sectionText()).toContain('CUDA out of memory');
    expect(buttonNamed('common.retry')?.disabled).toBe(false);

    // Retry is a new attempt: a new id, and retry keys that cannot collide with the failed attempt's.
    await act(() => userEvent.click(buttonNamed('common.retry')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);
    const requests = api.calibrateWorkingMemory.mock.calls.map(
      ([, request]) => request as { attemptId: string; idempotencyKey: string }
    );
    expect(new Set(requests.map((request) => request.attemptId)).size).toBe(1);
    expect(requests[0]!.attemptId).not.toBe('earlier-attempt');
    expect(requests.every((request) => request.idempotencyKey.includes(requests[0]!.attemptId))).toBe(true);
  });

  it('shows no notice where the method is verified and leaves runs without an attempt id on their own', async () => {
    serveSummary(
      summary({
        lines: [line('in_use')],
        runs: [run('legacy-1', 0, 'failed', 'old failure', null), run('legacy-2', 1, 'completed', null, null)],
      })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.stateInUse');

    expect(sectionText()).not.toContain('models.workingMemory.notVerified');
    // Only the newest legacy run is the latest attempt, so the older failure is not reported.
    expect(sectionText()).not.toContain('models.workingMemory.failed');
  });

  it('shows a skeleton while loading and a retryable error when a read fails', async () => {
    let resolvePlan: (plan: WorkingMemoryPlan) => void = () => undefined;
    api.getWorkingMemoryPlan.mockImplementationOnce(
      () =>
        new Promise<WorkingMemoryPlan>((resolve) => {
          resolvePlan = resolve;
        })
    );
    let failStatus: (error: Error) => void = () => undefined;
    api.getWorkingMemoryStatus.mockImplementationOnce(
      () =>
        new Promise<WorkingMemoryStatus>((_resolve, reject) => {
          failStatus = reject;
        })
    );
    await renderSection();

    // Opened for the first time, the panel waits for the plan.
    await expect.poll(() => host.querySelector('[aria-busy="true"]')).not.toBeNull();
    await act(() => {
      resolvePlan(supportedPlan());
      failStatus(new ApiError('{"detail":"store locked"}', 503));
    });
    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');
    expect(sectionText()).toContain('store locked');

    await act(() => userEvent.click(buttonNamed('common.retry')!));
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
  });
});

describe('WorkingMemorySection plan freshness', () => {
  /** One line instead of two, in fewer runs. */
  const LARGER = [planCase('fp16', 768)];
  const largerPlan = (): WorkingMemoryPlan => ({ ...supportedPlan(), cases: LARGER });
  /** The lines and runs of `CASES`, at other sizes. */
  const RESIZED = [planCase('fp16', 768), planCase('fp32', 768)];
  const resizedPlan = (): WorkingMemoryPlan => ({ ...supportedPlan(), cases: RESIZED });
  const FP32_ONLY = [planCase('fp32', 640)];

  /** Holds the first step's response, so the attempt stays queuing; returns its release. */
  const holdPost = () => {
    let release = () => undefined as void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const post = api.calibrateWorkingMemory.getMockImplementation()!;
    api.calibrateWorkingMemory.mockImplementationOnce(async (...args: unknown[]) => {
      await held;
      return post(...args);
    });
    return () => release();
  };

  const postedGraphs = () =>
    api.calibrateWorkingMemory.mock.calls.map(([, request]) => (request as { graph: { id: string } }).graph.id);

  const deferPlan = () => {
    let settle: { resolve: (plan: WorkingMemoryPlan) => void; reject: (error: Error) => void } = {
      reject: () => undefined,
      resolve: () => undefined,
    };
    api.getWorkingMemoryPlan.mockImplementationOnce(
      () =>
        new Promise<WorkingMemoryPlan>((resolve, reject) => {
          settle = { reject, resolve };
        })
    );
    return () => settle;
  };

  it('reads the plan again on every opening, keeping the panel on the plan it has meanwhile', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    expect(api.getWorkingMemoryPlan).toHaveBeenCalledTimes(1);

    await act(() => userEvent.click(sectionToggle()));
    const pending = deferPlan();
    await act(() => userEvent.click(sectionToggle()));

    await expect.poll(() => api.getWorkingMemoryPlan.mock.calls.length).toBe(2);
    // No skeleton while the second read runs: the panel keeps what it showed.
    expect(host.querySelector('[aria-busy="true"]')).toBeNull();
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);
    await act(() => pending().resolve(largerPlan()));
  });

  it('reads the plan again before a start and on confirmation, busy meanwhile, and runs the plan it read', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    // The server plans larger sizes from now on.
    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(largerPlan()));
    const pending = deferPlan();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')?.hasAttribute('data-loading')).toBe(true);
    expect(openDialog()).toBeNull();

    await act(() => pending().resolve(largerPlan()));
    await expect.poll(openDialog).not.toBeNull();
    expect(openDialog()!.textContent).toContain('models.workingMemory.size(height=768,width=768)');
    expect(openDialog()!.textContent).not.toContain('models.workingMemory.size(height=512,width=512)');
    expect(compile).toHaveBeenCalledTimes(1);
    expect(compile).toHaveBeenCalledWith(LARGER, expect.any(Promise), expect.any(AbortSignal));

    const confirming = deferPlan();
    const start = buttonNamed('models.workingMemory.calibrate', openDialog()!)!;
    // A second click before the first one rendered asks for no second read and starts no second attempt.
    await act(() => {
      start.click();
      start.click();
    });
    await expect.poll(() => start.hasAttribute('data-loading')).toBe(true);
    expect(api.getWorkingMemoryPlan).toHaveBeenCalledTimes(3);

    await act(() => confirming().resolve(largerPlan()));
    await expect.poll(openDialog).toBeNull();
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(3);
    expect(compile).toHaveBeenCalledTimes(2);
    expect(compile).toHaveBeenLastCalledWith(LARGER, expect.any(Promise), expect.any(AbortSignal));
  });

  it('starts new sizes read on confirmation without asking again, and names them in the progress', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    // A generation ended meanwhile: more memory is free, so the same lines run at larger sizes.
    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(resizedPlan()));
    const holdFirstPost = holdPost();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(openDialog).toBeNull();
    expect(compile).toHaveBeenCalledTimes(2);
    expect(compile).toHaveBeenLastCalledWith(RESIZED, expect.any(Promise), expect.any(AbortSignal));
    await expect.poll(sectionText).toContain('models.workingMemory.queuing(done=0,total=6)');
    expect(sectionText()).toContain(
      'models.workingMemory.pair(detail=fp16,label=models.workingMemory.size(height=768,width=768))'
    );

    await act(() => holdFirstPost());
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);
    expect(postedGraphs()).toEqual([
      RESIZED[0]!.id,
      RESIZED[0]!.id,
      RESIZED[0]!.id,
      RESIZED[1]!.id,
      RESIZED[1]!.id,
      RESIZED[1]!.id,
    ]);
  });

  it('compiles a confirmed calibration over the model record the server holds now, though its cases stayed', async () => {
    // The tab loaded its models before the section opened, as the app does on start.
    library.models = [mainModel('sd-1')];
    await refreshModels();
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    expect(getModelsSnapshot().modelsByKey.get(MODEL_KEY)?.base).toBe('sd-1');

    // Another tab changes the model's base while the dialog stands open. SD1 and SDXL decode alike, so the plan's
    // cases stay the same, but the queue runs the model as the record now says.
    library.models = [mainModel('sdxl')];
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);

    const bases = api.calibrateWorkingMemory.mock.calls.map(
      ([, request]) => (request as { graph: { base: string } }).graph.base
    );
    expect(new Set(bases)).toEqual(new Set(['sdxl']));
    // Each step states the model as that compile read it, which the server compares with the record it holds then.
    const identities = api.calibrateWorkingMemory.mock.calls.map(
      ([, request]) => (request as { graphModels: Record<string, { base: string }> }).graphModels
    );
    expect(identities).toEqual(
      identities.map(() => ({
        [MODEL_KEY]: { base: 'sdxl', format: 'checkpoint', hash: 'hash', type: 'main', variant: null },
      }))
    );
  });

  it('starts nothing when the plan read on confirmation is no longer supported, naming why', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    // Another process took the GPU's memory while the dialog stood open.
    api.getWorkingMemoryPlan.mockImplementationOnce(() =>
      Promise.resolve({ ...supportedPlan(), cases: [], reason: 'Only 3 GB of the GPU are free now.', supported: false })
    );
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(openDialog).toBeNull();
    await expect.poll(sectionText).toContain('Only 3 GB of the GPU are free now.');
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(true);
    expect(compile).toHaveBeenCalledTimes(1);
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    // Calibrate is disabled now; focus returns to the section's toggle rather than the page.
    await expect.poll(() => document.activeElement).toBe(sectionToggle());
  });

  it('shows runs read on confirmation that change what was confirmed, and starts them once confirmed again', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    expect(openDialog()!.textContent).not.toContain('models.workingMemory.dialogChanged');

    // fp32 no longer fits: the plan measures one line, in fewer runs.
    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(largerPlan()));
    const pending = deferPlan();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    // The busy button is disabled, which drops its focus.
    await expect.poll(() => document.activeElement).toBe(document.body);
    await act(() => pending().resolve(largerPlan()));

    await expect
      .poll(() => openDialog()?.querySelector('[role="alert"]')?.textContent)
      .toBe('models.workingMemory.dialogChanged');
    const dialog = openDialog()!;
    const notice = dialog.querySelector('[role="alert"]');
    expect(dialog.textContent).toContain('models.workingMemory.size(height=768,width=768)');
    expect(dialog.textContent).not.toContain('models.workingMemory.size(height=512,width=512)');
    expect(dialog.textContent).toContain('models.workingMemory.dialogRuns(count=3)');
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    // Focus stays on the action that confirms what the dialog shows now.
    await expect
      .poll(() => dialog.contains(document.activeElement) && document.activeElement?.textContent)
      .toBe('models.workingMemory.calibrate');

    // It changes once more before the next confirmation: the notice is a new one, which is announced again.
    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve({ ...supportedPlan(), cases: FP32_ONLY }));
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', dialog)!));
    await expect.poll(() => dialog.textContent).toContain('models.workingMemory.size(height=640,width=640)');
    const secondNotice = dialog.querySelector('[role="alert"]');
    expect(secondNotice?.textContent).toBe('models.workingMemory.dialogChanged');
    expect(secondNotice).not.toBe(notice);
    expect(notice?.isConnected).toBe(false);
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();

    // Confirmed as shown: it starts.
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', dialog)!));
    await expect.poll(openDialog).toBeNull();
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(3);
    expect(postedGraphs()).toEqual([FP32_ONLY[0]!.id, FP32_ONLY[0]!.id, FP32_ONLY[0]!.id]);
  });

  it('aborts the compiler and queues nothing when the dialog closes while new sizes compile', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(resizedPlan()));
    let compiling: AbortSignal | null = null;
    compile.mockImplementationOnce(
      (_cases, _models, signal) =>
        new Promise((_resolve, reject) => {
          compiling = signal;
          signal.addEventListener('abort', () => reject(new Error('Compiling was stopped.')), { once: true });
        })
    );
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => compile.mock.calls.length).toBe(2);

    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(openDialog).toBeNull();
    expect(compiling!.aborted).toBe(true);
    // The aborted compile is no reason to show: Calibrate is available as before.
    expect(sectionText()).not.toContain('models.workingMemory.blockedCompile');
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
  });

  it('starts nothing when the dialog closes while the plan is read on confirmation', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    const opener = buttonNamed('models.workingMemory.calibrate')!;
    await act(() => userEvent.click(opener));
    await expect.poll(openDialog).not.toBeNull();

    const pending = deferPlan();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect
      .poll(() => buttonNamed('models.workingMemory.calibrate', openDialog()!)?.hasAttribute('data-loading'))
      .toBe(true);
    await act(() => userEvent.click(buttonNamed('common.cancel', openDialog()!)!));
    await expect.poll(openDialog).toBeNull();
    await expect.poll(() => document.activeElement).toBe(opener);

    await act(() => pending().resolve(supportedPlan()));
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();

    // The next opening compiles and confirms afresh.
    await act(() => userEvent.click(opener));
    await expect.poll(openDialog).not.toBeNull();
    expect(compile).toHaveBeenCalledTimes(2);
    const start = buttonNamed('models.workingMemory.calibrate', openDialog()!)!;
    expect(start.hasAttribute('data-loading')).toBe(false);
    await act(() => userEvent.click(start));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);
  });

  it('starts a confirmed calibration whose plan still holds after the section unmounted during the read', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    const pending = deferPlan();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    // Leaving the model's details does not take the confirmation back; only closing the dialog does.
    await act(() => root.unmount());
    await act(() => pending().resolve(supportedPlan()));

    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);
    expect(compile).toHaveBeenCalledTimes(2);
    expect(compile).toHaveBeenLastCalledWith(CASES, expect.any(Promise), expect.any(AbortSignal));
  });

  it('shows a plan read that fails on confirmation as an error with a retry, starting nothing', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    api.getWorkingMemoryPlan.mockImplementationOnce(() => Promise.reject(new ApiError('{"detail":"GPU lost"}', 503)));
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');
    expect(sectionText()).toContain('GPU lost');
    await expect.poll(openDialog).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    // The panel holding the dialog's opener is gone; focus returns to the section's toggle rather than the page.
    await expect.poll(() => document.activeElement).toBe(sectionToggle());
    await act(() => userEvent.click(buttonNamed('common.retry')!));
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);
  });

  /** Another tab's attempt of the model, two of six steps done. */
  const theirs = () =>
    summary({
      runs: [1, 2, 3, 4, 5, 6].map((number) =>
        run(`theirs-${String(number)}`, number, number <= 2 ? 'completed' : 'running', null, 'theirs', {
          attempt_steps: 6,
          held_back: 'running',
        })
      ),
    });

  /**
   * The GPU's status, a run of `modelKey` running on it: the server takes one attempt per GPU. With `isCurrent` false
   * it runs on another device class.
   */
  const gpuBusyWith = (modelKey: string, { isCurrent = true }: { isCurrent?: boolean } = {}): WorkingMemoryStatus =>
    status({
      device_classes: [
        {
          device_class: {
            arch: null,
            backend: 'cuda',
            id: DEVICE.id,
            name: DEVICE.name,
            total_mib: 24_576,
            vendor: 'x',
          },
          device_class_id: DEVICE.id,
          envs: [],
          is_current: isCurrent,
          measurement_method: 'allocator',
          measurement_verified: true,
          point_counts: [],
          problem: null,
          runs: [
            {
              attempt_id: 'theirs',
              attempt_steps: 6,
              created_at: new Date(T0).toISOString(),
              env_id: 'env',
              error: null,
              error_code: null,
              item_ids: [1],
              model_hash: null,
              model_key: modelKey,
              point_count: 0,
              run_id: 'theirs-1',
              status: 'running',
              user_id: 'other',
            },
          ],
          usable: true,
        },
      ],
    });

  it('starts nothing when another tab started a calibration on this GPU while the dialog stood open', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    // Another model's calibration took the GPU: this section's summary cannot know, only a fresh status read can.
    api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(gpuBusyWith('other-model')));
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(openDialog).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    // Beside Calibrate, naming the model that holds the GPU; no toast, which only the server's refusal raises.
    await expect.poll(sectionText).toContain('models.workingMemory.busyOnGpu(model=other-model)');
    expect(notify.info).not.toHaveBeenCalled();
    expect(notify.error).not.toHaveBeenCalled();
    // Nothing of this model runs: Calibrate stays available, and focus returns to it.
    expect(sectionText()).not.toContain('models.workingMemory.failed');
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.calibrate');
  });

  it('names the model that holds the GPU when Calibrate finds it busy, and keeps focus on Calibrate', async () => {
    library.models = [mainModel('sdxl'), { ...mainModel('sd-1'), key: 'other-model', name: 'Other SD1' }];
    await refreshModels();
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    // The status read right before the start answers late, so the busy Calibrate has dropped its focus by then.
    let answer = () => undefined as void;
    api.getWorkingMemoryStatus.mockImplementation(
      () =>
        new Promise<WorkingMemoryStatus>((resolve) => {
          answer = () => resolve(gpuBusyWith('other-model'));
        })
    );

    const opener = buttonNamed('models.workingMemory.calibrate')!;
    await act(() => userEvent.click(opener));
    await expect.poll(() => document.activeElement).toBe(document.body);
    await act(() => answer());

    const reason = 'models.workingMemory.busyOnGpu(model=Other SD1)';
    await expect.poll(sectionText).toContain(reason);
    expect(openDialog()).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    expect(notify.info).not.toHaveBeenCalled();
    expect(opener.disabled).toBe(false);
    // Announced politely: a wait, not an error.
    const described = document.getElementById(opener.getAttribute('aria-describedby')!);
    expect(described?.textContent).toBe(reason);
    expect(described?.getAttribute('role')).toBe('status');
    await expect.poll(() => document.activeElement).toBe(opener);
  });

  it('shows this model’s own attempt with Stop when Calibrate finds it running, focusing Stop', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    serveSummary(theirs());
    api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(gpuBusyWith(MODEL_KEY)));

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));

    await expect.poll(sectionText).toContain('models.workingMemory.running(done=2,total=6)');
    expect(openDialog()).toBeNull();
    expect(sectionText()).not.toContain('models.workingMemory.busyOnGpu');
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.stop');
  });

  it('starts while another device class calibrates: the GPU this section names is free', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    api.getWorkingMemoryStatus.mockImplementation(() =>
      Promise.resolve(gpuBusyWith('other-model', { isCurrent: false }))
    );

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);
    expect(sectionText()).not.toContain('models.workingMemory.busyOnGpu');
  });

  it('starts nothing when the status cannot be read right before the start, offering a retry', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    api.getWorkingMemoryStatus.mockImplementation(() =>
      Promise.reject(new ApiError('{"detail":"The calibration store is locked"}', 503))
    );
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');
    expect(sectionText()).toContain('The calibration store is locked');
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    expect(buttonNamed('common.retry')).toBeDefined();
  });

  it('says the model changed when the server refuses a later step for it, in its own words', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    const post = api.calibrateWorkingMemory.getMockImplementation()!;
    const serverText = 'This SDXL changed after this calibration was planned (base). Calibrate again.';
    api.calibrateWorkingMemory.mockImplementation((...args: unknown[]) =>
      api.calibrateWorkingMemory.mock.calls.length === 3
        ? Promise.reject(new ApiError(JSON.stringify({ detail: { code: 'model_changed', message: serverText } }), 409))
        : post(...args)
    );

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(3);

    // The server lists the two runs queued before the refusal.
    const attemptId = (api.calibrateWorkingMemory.mock.calls[0]![1] as { attemptId: string }).attemptId;
    serveSummary(
      summary({
        runs: [1, 2].map((number) =>
          run(`run-${String(number)}`, number, 'completed', null, attemptId, { ...ABORTED, attempt_steps: 6 })
        ),
      })
    );
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.partial');
    expect(sectionText()).toContain('models.workingMemory.modelChanged');
    expect(sectionText()).not.toContain(serverText);
  });

  it('stops posting when another calibration ran on the GPU between its steps, saying so in its own words', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    const post = api.calibrateWorkingMemory.getMockImplementation()!;
    const serverText =
      'Another calibration ran on this GPU after this calibration started, so its measurements can no longer be combined. Calibrate again.';
    api.calibrateWorkingMemory.mockImplementation((...args: unknown[]) =>
      api.calibrateWorkingMemory.mock.calls.length === 2
        ? Promise.reject(
            new ApiError(JSON.stringify({ detail: { code: 'calibration_interleaved', message: serverText } }), 409)
          )
        : post(...args)
    );

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(2);

    // The server lists the one run queued before the refusal, held back with its short attempt.
    const attemptId = (api.calibrateWorkingMemory.mock.calls[0]![1] as { attemptId: string }).attemptId;
    serveSummary(summary({ runs: [run('run-1', 1, 'completed', null, attemptId, { ...ABORTED, attempt_steps: 6 })] }));
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.interleaved');
    expect(sectionText()).toContain('models.workingMemory.discarded');
    expect(sectionText()).not.toContain(serverText);
    expect(buttonNamed('common.retry')).toBeDefined();
    // The refused step ends the attempt: no later step is posted.
    expect(api.calibrateWorkingMemory).toHaveBeenCalledTimes(2);
    expect(notify.info).not.toHaveBeenCalled();
  });

  it('follows an attempt of this model another tab started, found right before the start', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    serveSummary(theirs());
    api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(gpuBusyWith(MODEL_KEY)));
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(openDialog).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    expect(notify.info).not.toHaveBeenCalled();
    await expect.poll(sectionText).toContain('models.workingMemory.running(done=2,total=6)');
    // Its own attempt shows as running, not as a reason naming it.
    expect(sectionText()).not.toContain('models.workingMemory.busyOnGpu');
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.stop');
  });

  it('says the model changed when its queued steps failed for it, offering to calibrate again', async () => {
    // The record changed after queuing: the server failed the items as it admitted them.
    const changed = { attempt_steps: 6, error_code: 'model_changed' as const };
    serveSummary(
      summary({
        runs: [1, 2, 3, 4, 5, 6].map((number) =>
          run(
            `run-${String(number)}`,
            number,
            'failed',
            'This SDXL changed after this calibration was planned (hash). Calibrate again.',
            'attempt',
            changed
          )
        ),
      })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.failed');

    expect(sectionText()).toContain('models.workingMemory.modelChanged');
    expect(sectionText()).not.toContain('This SDXL changed after this calibration was planned');
    expect(buttonNamed('common.retry')?.disabled).toBe(false);
  });

  it('follows the attempt the server says is already running, reading its refusal as no failure', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    // Another tab started one between the fresh read and the first step: the server refuses this attempt.
    api.calibrateWorkingMemory.mockImplementationOnce(() => {
      serveSummary(theirs());
      return Promise.reject(
        new ApiError(
          '{"detail":{"code":"calibration_running","message":"Another calibration is still running on this GPU. Wait until it has finished, or stop it, then calibrate again."}}',
          409
        )
      );
    });

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(() => notify.info.mock.calls).toEqual([['models.workingMemory.alreadyRunning']]);
    expect(api.calibrateWorkingMemory).toHaveBeenCalledTimes(1);
    await expect.poll(sectionText).toContain('models.workingMemory.running(done=2,total=6)');
    expect(sectionText()).not.toContain('models.workingMemory.failed');
    expect(sectionText()).not.toContain('models.workingMemory.partial');
  });

  it('says the model changed when the server refuses a step for it, offering to calibrate again', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    api.calibrateWorkingMemory.mockImplementationOnce(() =>
      Promise.reject(
        new ApiError(
          '{"detail":{"code":"model_changed","message":"This SDXL changed after this calibration was planned (hash). Calibrate again."}}',
          409
        )
      )
    );

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(sectionText).toContain('models.workingMemory.modelChanged');
    expect(sectionText()).toContain('models.workingMemory.failed');
    expect(api.calibrateWorkingMemory).toHaveBeenCalledTimes(1);
    expect(buttonNamed('common.retry')?.disabled).toBe(false);
  });

  it('opens nothing when the model records cannot be read, naming why, and checks again on the next click', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('neverCalibrated');
    // Compiling over the records the tab loaded earlier could build for a model the server has changed since.
    library.failNext = new ApiError('{"detail":"The model records cannot be read"}', 503);

    const opener = buttonNamed('models.workingMemory.calibrate')!;
    await act(() => userEvent.click(opener));

    await expect
      .poll(sectionText)
      .toContain('models.workingMemory.blockedCompile(problem=The model records cannot be read)');
    expect(openDialog()).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    expect(opener.disabled).toBe(false);

    await act(() => userEvent.click(opener));
    await expect.poll(openDialog).not.toBeNull();
    expect(sectionText()).not.toContain('models.workingMemory.blockedCompile');
  });

  it('opens nothing for a plan that does not compile here, naming why, and checks again on the next click', async () => {
    // The server plans a FLUX.1 decode; Generate cannot build it without a T5 encoder installed.
    const flux = {
      ...planCase('bf16', 768),
      estimator: 'flux_vae_decode',
      id: 'flux_vae_decode-decode-bf16-768x768',
      settings: {},
    };
    api.getWorkingMemoryPlan.mockImplementation(() =>
      Promise.resolve({ ...supportedPlan(), cases: [flux], estimators: ['flux_vae_decode'] })
    );
    const missing = `Case ${flux.id}: FLUX needs a T5 encoder; install one.`;
    let refuse = () => undefined as void;
    compile.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          refuse = () => reject(new Error(missing));
        })
    );
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');

    const opener = buttonNamed('models.workingMemory.calibrate')!;
    await act(() => userEvent.click(opener));
    // Busy while it compiles, which drops the button's focus.
    await expect.poll(() => document.activeElement).toBe(document.body);
    await act(() => refuse());

    const reason = `models.workingMemory.blockedCompile(problem=${missing})`;
    await expect.poll(sectionText).toContain(reason);
    const alert = host.querySelector('[role="alert"]')!;
    expect(alert.textContent).toBe(reason);
    expect(openDialog()).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    // Installing the encoder happens elsewhere: Calibrate stays available and names the reason.
    expect(opener.disabled).toBe(false);
    expect(document.getElementById(opener.getAttribute('aria-describedby')!)).toBe(alert);
    await expect.poll(() => document.activeElement).toBe(opener);

    await act(() => userEvent.click(opener));
    await expect.poll(openDialog).not.toBeNull();
    expect(sectionText()).not.toContain('models.workingMemory.blockedCompile');
  });

  it('closes the dialog naming why the plan read on confirmation does not compile, starting nothing', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();

    // New sizes are compiled again, and the encoder they need was removed meanwhile.
    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(resizedPlan()));
    compile.mockImplementationOnce(() => Promise.reject(new Error('The text encoder is not installed.')));
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(openDialog).toBeNull();
    await expect
      .poll(sectionText)
      .toContain('models.workingMemory.blockedCompile(problem=The text encoder is not installed.)');
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.calibrate');
  });

  it('starts nothing when the plan read before a start is no longer supported, naming why', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    const pending = deferPlan();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(() => document.activeElement).toBe(document.body);
    await act(() =>
      pending().resolve({
        ...supportedPlan(),
        cases: [],
        reason: 'Only 3 GB of the GPU are free now.',
        supported: false,
      })
    );

    await expect.poll(sectionText).toContain('Only 3 GB of the GPU are free now.');
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(true);
    expect(openDialog()).toBeNull();
    expect(compile).not.toHaveBeenCalled();
    // Calibrate is disabled now; focus returns to the section's toggle rather than the page.
    await expect.poll(() => document.activeElement).toBe(sectionToggle());
  });

  it('shows a plan read that fails before a start as an error with a retry', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
    const pending = deferPlan();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(() => document.activeElement).toBe(document.body);
    await act(() => pending().reject(new ApiError('{"detail":"GPU lost"}', 503)));

    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');
    expect(sectionText()).toContain('GPU lost');
    expect(openDialog()).toBeNull();
    // The panel holding Calibrate is gone; focus returns to the section's toggle rather than the page.
    await expect.poll(() => document.activeElement).toBe(sectionToggle());
    await act(() => userEvent.click(buttonNamed('common.retry')!));
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);
  });
});
