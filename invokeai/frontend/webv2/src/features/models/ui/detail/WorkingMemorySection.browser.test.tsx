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
import { WORKING_MEMORY_POLL_MS } from '@features/models/data/workingMemoryQueries';
import { CalibrationGraphCompilerProvider } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, Activity, useState, useCallback } from 'react';
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

const library = vi.hoisted(() => ({ failNext: null as Error | null, models: [] as unknown[] }));

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
  point_count: runStatus === 'running' ? 0 : 1,
  run_id: id,
  status: runStatus,
  ...overrides,
});

const ABORTED = { held_back: 'aborted_attempt' } as const;

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

const RetainedSection = () => {
  const [visible, setVisible] = useState(true);
  const toggleVisible = useCallback(() => setVisible((value) => !value), []);
  return (
    <>
      <button type="button" data-test-visibility="" onClick={toggleVisible}>
        Toggle visibility
      </button>
      <Activity mode={visible ? 'visible' : 'hidden'}>
        <WorkingMemorySection modelKey={MODEL_KEY} />
      </Activity>
    </>
  );
};

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
            <RetainedSection />
          </CalibrationGraphCompilerProvider>
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

  if (expanded) {
    await act(() => userEvent.click(sectionToggle()));
  }
};

const sectionToggle = () => host.querySelector<HTMLButtonElement>('h3 button')!;

const buttonNamed = (name: string, within: ParentNode = document): HTMLButtonElement | undefined =>
  [...within.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent.trim() === name);

const openDialog = () => document.querySelector<HTMLElement>('[role="dialog"][data-state="open"]');
const openAlertDialog = () => document.querySelector<HTMLElement>('[role="alertdialog"][data-state="open"]');
const sectionText = () => host.textContent;

const serveSummary = (next: WorkingMemorySummary | Error) => {
  currentSummary = next;
};

const refreshSummary = () => act(() => queryClient.invalidateQueries());

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
    const enqueue = api.calibrateWorkingMemory.getMockImplementation()!;
    let releaseFirst: () => void = () => {};
    api.calibrateWorkingMemory.mockImplementationOnce(
      (...args) =>
        new Promise((resolve) => {
          releaseFirst = () => {
            void Promise.resolve(enqueue(...args)).then(resolve);
          };
        })
    );
    // The summary polls on an interval while a run is running: the test steps that interval rather than wait it out.
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');

    const section = host.querySelector('section')!;
    expect(section.getAttribute('aria-labelledby')).toBe(host.querySelector('h3')!.id);
    expect(host.querySelector('h3')!.textContent).toBe('models.workingMemory.title');
    expect(sectionText()).toContain(DEVICE.name);
    expect(sectionText()).toContain('models.workingMemory.stateNone');
    expect(compile).not.toHaveBeenCalled();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
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
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(1);
    await act(() => userEvent.click(host.querySelector<HTMLButtonElement>('[data-test-visibility]')!));
    await act(async () => {
      releaseFirst();
      await Promise.resolve();
    });
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);

    expect(
      queryClient
        .getQueryCache()
        .getAll()
        .filter((query) => query.queryKey.includes('workingMemory'))
        .every((query) => query.getObserversCount() === 0)
    ).toBe(true);
    await act(() => userEvent.click(host.querySelector<HTMLButtonElement>('[data-test-visibility]')!));

    expect(compile).toHaveBeenCalledTimes(2);
    expect(compile).toHaveBeenLastCalledWith(CASES, expect.any(Promise), expect.any(AbortSignal));
    const requests = api.calibrateWorkingMemory.mock.calls.map(
      ([key, request]) => [key, (request as { graph: { id: string } }).graph.id] as const
    );
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
    const reset = buttonNamed('common.reset')!;
    expect(reset.disabled).toBe(true);
    expect(document.getElementById(reset.getAttribute('aria-describedby')!)?.textContent).toBe(
      'models.workingMemory.blockedRunning'
    );

    serveSummary(
      summary({
        lines: [line('in_use')],
        runs: [
          run('old', -600, 'failed', 'old failure'),
          ...[1, 2, 3, 4, 5, 6].map((number) => run(`run-${String(number)}`, number, 'completed', null, attemptId)),
        ],
      })
    );
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
    const alignments = (selector: string) =>
      [...table.querySelectorAll(selector)].map((cell) => getComputedStyle(cell).textAlign);
    expect(alignments('thead th')).toEqual(['start', 'end', 'end', 'end']);
    expect(alignments('tbody tr:first-child > *')).toEqual(['start', 'end', 'end', 'end']);
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
    const describedBy = openAlertDialog()!.getAttribute('aria-describedby') ?? '';
    expect(document.getElementById(describedBy)?.textContent).toBe('models.workingMemory.stopBody');
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
    await expect.poll(() => document.activeElement?.textContent).toBe('common.retry');
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

    const unreachable = () => Promise.reject(new TypeError('Failed to fetch'));
    api.getWorkingMemoryPlan.mockImplementation(unreachable);
    api.getWorkingMemoryStatus.mockImplementation(unreachable);
    serveSummary(new TypeError('Failed to fetch'));
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');

    api.getWorkingMemoryPlan.mockImplementation(() => Promise.resolve(supportedPlan()));
    api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(status()));
    serveSummary(finished);
    await refreshSummary();
    await expect.poll(() => buttonNamed('models.workingMemory.recalibrate')).toBeDefined();
    expect(sectionText()).not.toContain('models.workingMemory.failed');
    expect(sectionText()).not.toContain('Failed to fetch');
  });

  it('starts collapsed, keeps its panel out of the tab order, and asks for the plan only once opened', async () => {
    serveSummary(summary({ lines: [line('in_use')], runs: [run('run-1', 0, 'completed')] }));
    await renderSection({ expanded: false });

    const toggle = sectionToggle();
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    const panel = document.getElementById(toggle.getAttribute('aria-controls')!)!;
    expect(panel.hidden).toBe(true);
    expect(buttonNamed('models.workingMemory.recalibrate')).toBeUndefined();
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

  it('resets after confirmation and restores focus', async () => {
    serveSummary(summary({ lines: [line('in_use')], runs: [run('run-1', 0, 'completed')] }));
    await renderSection();
    await expect.poll(() => buttonNamed('common.reset')).toBeDefined();

    await act(() => userEvent.click(buttonNamed('common.reset')!));
    await expect.poll(openAlertDialog).not.toBeNull();
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
  const LARGER = [planCase('fp16', 768)];
  const largerPlan = (): WorkingMemoryPlan => ({ ...supportedPlan(), cases: LARGER });
  const RESIZED = [planCase('fp16', 768), planCase('fp32', 768)];
  const resizedPlan = (): WorkingMemoryPlan => ({ ...supportedPlan(), cases: RESIZED });
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

  it('reads the plan again before a start and on confirmation, busy meanwhile, and runs the plan it read', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
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
    expect(sectionText()).not.toContain('models.workingMemory.blockedCompile');
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
  });

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

    api.getWorkingMemoryStatus.mockImplementation(() => Promise.resolve(gpuBusyWith('other-model')));
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));

    await expect.poll(openDialog).toBeNull();
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
    await expect.poll(sectionText).toContain('models.workingMemory.busyOnGpu(model=other-model)');
    expect(notify.info).not.toHaveBeenCalled();
    expect(notify.error).not.toHaveBeenCalled();
    expect(sectionText()).not.toContain('models.workingMemory.failed');
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.calibrate');
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

    const attemptId = (api.calibrateWorkingMemory.mock.calls[0]![1] as { attemptId: string }).attemptId;
    serveSummary(summary({ runs: [run('run-1', 1, 'completed', null, attemptId, { ...ABORTED, attempt_steps: 6 })] }));
    await refreshSummary();
    await expect.poll(sectionText).toContain('models.workingMemory.interleaved');
    expect(sectionText()).toContain('models.workingMemory.discarded');
    expect(sectionText()).not.toContain(serverText);
    expect(buttonNamed('common.retry')).toBeDefined();
    expect(api.calibrateWorkingMemory).toHaveBeenCalledTimes(2);
    expect(notify.info).not.toHaveBeenCalled();
  });
});
