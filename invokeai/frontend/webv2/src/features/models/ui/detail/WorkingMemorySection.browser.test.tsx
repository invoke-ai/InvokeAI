import type {
  CalibrationGraphCompiler,
  WorkingMemoryCase,
  WorkingMemoryLine,
  WorkingMemoryPlan,
  WorkingMemoryRun,
  WorkingMemoryStatus,
  WorkingMemorySummary,
} from '@features/models/core/workingMemory';

import { ChakraProvider } from '@chakra-ui/react';
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
  attemptId: string | null = 'earlier-attempt'
): WorkingMemoryRun => ({
  attempt_id: attemptId,
  created_at: new Date(T0 + offset * 1_000).toISOString(),
  error,
  point_count: runStatus === 'completed' ? 1 : 0,
  run_id: id,
  status: runStatus,
});

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

const renderSection = async () => {
  host = document.createElement('div');
  host.style.width = '720px';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { WorkingMemorySection } = await import('./WorkingMemorySection');

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
};

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

beforeEach(() => {
  document.documentElement.lang = 'en';
  accountLifecycle.activate('wm-section-test', ':user:wm-section-test');
  compile = vi.fn<CalibrationGraphCompiler>((cases) =>
    Promise.resolve(cases.map((item) => ({ caseId: item.id, data: [], graph: { edges: [], id: item.id, nodes: {} } })))
  );
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
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
  vi.clearAllMocks();
});

describe('WorkingMemorySection', () => {
  it('calibrates a model that was never calibrated: dialog, queued runs, progress, then the measurements', async () => {
    await renderSection();
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');

    const section = host.querySelector('section')!;
    expect(section.getAttribute('aria-labelledby')).toBe(host.querySelector('h3')!.id);
    expect(host.querySelector('h3')!.textContent).toBe('models.workingMemory.title');
    expect(sectionText()).toContain(DEVICE.name);
    expect(sectionText()).toContain('models.workingMemory.stateNone');
    // The compiler loads only once a calibration starts.
    expect(compile).not.toHaveBeenCalled();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    const dialog = openDialog()!;
    expect(dialog.textContent).toContain('models.workingMemory.dialogDuration(minutes=3)');
    expect(dialog.textContent).toContain('models.workingMemory.dialogQueueNote');
    expect(dialog.textContent).toContain(
      'models.workingMemory.pair(detail=fp16,label=models.workingMemory.size(height=512,width=512))'
    );
    expect(dialog.textContent).toContain(
      'models.workingMemory.pair(detail=fp32,label=models.workingMemory.size(height=512,width=512))'
    );
    expect(dialog.textContent).toContain('models.workingMemory.dialogRuns(count=6)');
    expect(dialog.textContent).toContain('Shared SD1');
    expect(compile).not.toHaveBeenCalled();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => api.calibrateWorkingMemory.mock.calls.length).toBe(6);

    expect(compile).toHaveBeenCalledTimes(1);
    expect(compile).toHaveBeenCalledWith(CASES, expect.any(AbortSignal));
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
    // Polling picks the end up without any other trigger.
    await expect.poll(() => host.querySelector('table'), { timeout: 5_000 }).not.toBeNull();

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
    expect(compile).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{Enter}'));
    await expect
      .poll(() => openDialog()?.contains(document.activeElement) && document.activeElement?.textContent)
      .toBe('models.workingMemory.calibrate');
    // Hold the compiler so the attempt stays in its running state while focus settles.
    let releaseCompile = () => undefined as void;
    const compiled = new Promise<void>((resolve) => {
      releaseCompile = resolve;
    });
    const compileCases = compile.getMockImplementation()!;
    compile.mockImplementationOnce(async (cases) => {
      await compiled;
      return compileCases(cases, new AbortController().signal);
    });
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.poll(openDialog).toBeNull();
    // Starting disables the opener; focus moves to the action that is available now instead of the page.
    await expect.poll(() => document.activeElement?.textContent).toBe('models.workingMemory.stop');
    expect(host.contains(document.activeElement)).toBe(true);

    await act(() => releaseCompile());
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
    expect(openAlertDialog()!.textContent).toContain('models.workingMemory.stopBody');
    expect(api.cancelCalibrationRun).not.toHaveBeenCalled();

    api.cancelCalibrationRun.mockImplementation(() => {
      serveSummary(
        summary({ runs: [run('run-1', 0, 'completed'), run('run-2', 1, 'partial'), run('run-3', 2, 'canceled')] })
      );
      return Promise.resolve();
    });
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop', openAlertDialog()!)!));
    await expect.poll(openAlertDialog).toBeNull();

    expect(api.cancelCalibrationRun.mock.calls.map(([runId]) => runId)).toEqual(['run-3', 'run-2']);
    await expect.poll(sectionText).toContain('models.workingMemory.partial');
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
      summary({ runs: [1, 2, 3].map((number) => run(`run-${String(number)}`, number, 'completed', null, attemptId)) })
    );
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop', openAlertDialog()!)!));
    await act(() => releasePost());
    await expect.poll(openAlertDialog).toBeNull();

    expect(api.calibrateWorkingMemory).toHaveBeenCalledTimes(3);
    await expect.poll(sectionText).toContain('models.workingMemory.partialCanceled');
    expect(sectionText()).not.toContain('models.workingMemory.partialQueued');
  });

  it('stops an attempt that is still compiling: the compiler is aborted and nothing is queued', async () => {
    compile.mockImplementationOnce(
      (_cases, signal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
        })
    );
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.calibrate')).toBeDefined();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(() => buttonNamed('models.workingMemory.stop')).toBeDefined();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop')!));
    await expect.poll(openAlertDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.stop', openAlertDialog()!)!));

    // The attempt left queuing: it reads as stopped, not as a failure, and Calibrate is available again.
    await expect.poll(sectionText).toContain('models.workingMemory.canceled');
    expect(sectionText()).not.toContain('models.workingMemory.failed');
    expect(buttonNamed('models.workingMemory.stop', host)).toBeUndefined();
    expect(buttonNamed('models.workingMemory.calibrate')?.disabled).toBe(false);
    expect(api.calibrateWorkingMemory).not.toHaveBeenCalled();
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

  it('lets an attempt that could not reach the server give way to one that ran after it', async () => {
    const earlier = run('old', 0, 'completed', null, 'old');
    serveSummary(summary({ lines: [line('in_use')], runs: [earlier] }));
    compile.mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')));
    await renderSection();
    await expect.poll(() => buttonNamed('models.workingMemory.recalibrate')).toBeDefined();

    await act(() => userEvent.click(buttonNamed('models.workingMemory.recalibrate')!));
    await expect.poll(openDialog).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.workingMemory.calibrate', openDialog()!)!));
    await expect.poll(sectionText).toContain('models.workingMemory.failed');
    expect(sectionText()).toContain('Failed to fetch');

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
    expect(sectionText()).not.toContain('Failed to fetch');
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
        ],
        runs: [run('run-1', 0, 'completed')],
      })
    );
    await renderSection();
    await expect.poll(() => host.querySelectorAll('table').length).toBe(2);

    const [plain, mixed] = [...host.querySelectorAll('table')];
    const [first, , excluded] = [...mixed!.querySelectorAll('tbody tr')];
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
    expect(document.activeElement?.textContent).toBe('Cancel');

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
    api.getWorkingMemoryStatus.mockImplementationOnce(() =>
      Promise.reject(new ApiError('{"detail":"store locked"}', 503))
    );
    await renderSection();

    expect(host.querySelector('[aria-busy="true"]')).not.toBeNull();
    await act(() => resolvePlan(supportedPlan()));
    await expect.poll(sectionText).toContain('models.workingMemory.loadFailed');
    expect(sectionText()).toContain('store locked');

    await act(() => userEvent.click(buttonNamed('common.retry')!));
    await expect.poll(sectionText).toContain('models.workingMemory.neverCalibrated');
  });
});
