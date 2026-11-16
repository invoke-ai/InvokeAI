/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateWidgetValues, MainModelConfig } from '@features/generation/contracts';
import type * as ModelsModule from '@features/models';
import type { QueueItemReadModel } from '@features/queue/contracts';
import type * as QueuePublicApi from '@features/queue/publicApi';
import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/settings';
import { buildQueueItemOrigin } from '@features/queue/contracts';
import { loadQueueWidgetImplementation } from '@features/queue/widget';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { QueueUiAdapterProvider } from './QueueUiAdapter';

/**
 * Queue entry recall end to end: the Queue widget's real rows, menus and details, the App adapter that wires the
 * Workbench's recall into them, a real Workbench store and the real hotkey runtime. Only context plumbing, notices,
 * panel opening and the model catalog are replaced.
 */

const mocks = vi.hoisted(() => ({
  extensions: null as unknown as ExtensionRegistry,
  notify: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  openWidget: vi.fn(),
  main: { base: 'sdxl', key: 'main', name: 'SDXL', type: 'main' } as MainModelConfig & ModelsModule.ModelConfig,
  store: null as unknown as ReturnType<typeof createWorkbenchStore>,
}));
const { main } = mocks;

vi.mock('@features/models', async (importOriginal) => {
  const original = await importOriginal<typeof ModelsModule>();
  const snapshot = { ...original.getModelsSnapshot(), models: [mocks.main], status: 'loaded' as const };

  return {
    ...original,
    ensureModelsLoaded: () => Promise.resolve(),
    getModelsSnapshot: () => snapshot,
    useModelsSelector: <Selected,>(selector: (state: typeof snapshot) => Selected) => selector(snapshot),
  };
});
vi.mock('@features/queue/publicApi', async (importOriginal) => ({
  ...(await importOriginal<typeof QueuePublicApi>()),
  getQueueItemDetailOptions: (itemId: number) => ({ queryFn: () => null, queryKey: ['queue', 'item-detail', itemId] }),
  getQueueReadModelOptions: () => ({ queryFn: () => null, queryKey: ['queue'], staleTime: Number.POSITIVE_INFINITY }),
}));
// The first-party commands need the whole application; the Gallery's recall stands in for them below.
vi.mock('@workbench/hotkeys/firstPartyCommands', () => ({ useRegisterFirstPartyCommands: () => {} }));
vi.mock('@workbench/useNotify', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNotify: () => mocks.notify,
}));
vi.mock('@workbench/useOpenWorkbenchWidget', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useOpenWorkbenchWidget: () => mocks.openWidget,
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => {
  const useStoreSelector = <Selected,>(
    selector: (snapshot: WorkbenchSnapshot) => Selected,
    isEqual: (left: Selected, right: Selected) => boolean = shallowEqual
  ) => useExternalStoreSelector(mocks.store.subscribe, mocks.store.getSnapshot, selector, isEqual);

  return {
    ...(await importOriginal<typeof WorkbenchContextModule>()),
    useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected, isEqual?: never) =>
      useStoreSelector((snapshot) => selector(snapshot.activeProject), isEqual),
    useOptionalWorkbenchExtensions: () => mocks.extensions,
    useWidgetValuesSelector: <Selected,>(
      widgetId: string,
      selector: (values: Record<string, unknown>) => Selected,
      isEqual?: never
    ) =>
      useStoreSelector(
        (snapshot) =>
          selector(getResolvedProjectWidgetValues(snapshot.activeProject, widgetId, snapshot.scopedDefaults)),
        isEqual
      ),
    useWorkbenchCommands: () => mocks.store.commands,
    useWorkbenchExtensions: () => mocks.extensions,
    useWorkbenchInternalStore: () => mocks.store,
    useWorkbenchLiveCanvasEngines: () => ({ commitFrame: () => false }),
    useWorkbenchQueries: () => mocks.store.queries,
    useWorkbenchSelector: useStoreSelector,
    useWorkbenchSubscription: () => mocks.store.subscribe,
  };
});

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

seedArchitectureCapabilities();

const submitted = (positivePrompt: string): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(main),
  model: main,
  positivePrompt,
  steps: 33,
});

/** A run this tab submitted and the backend accepted, with the snapshot its recall restores. */
const localRun = (id: string, positivePrompt: string): Project['queue']['items'][number] =>
  ({
    backendItemIds: [1],
    cancellable: false,
    id,
    snapshot: {
      presentation: { batchCount: 1, height: 1024, positivePrompt, width: 1024 },
      recall: { generateValues: submitted(positivePrompt) },
      sourceId: 'generate',
      submittedAt: '2026-01-01T00:00:00.000Z',
    },
    status: 'completed',
  }) as unknown as Project['queue']['items'][number];

/** A workflow run this tab submitted: its snapshot records no Generate settings. */
const localWorkflowRun = (id: string): Project['queue']['items'][number] =>
  ({
    backendItemIds: [5],
    cancellable: false,
    id,
    snapshot: { presentation: { batchCount: 1 }, sourceId: 'workflow', submittedAt: '2026-01-01T00:00:00.000Z' },
    status: 'completed',
  }) as unknown as Project['queue']['items'][number];

const backendItem = (overrides: Partial<QueueItemReadModel> & Pick<QueueItemReadModel, 'id'>): QueueItemReadModel => ({
  batchId: `batch-${overrides.id}`,
  createdAt: '2026-01-01T00:00:00Z',
  fieldValues: [{ fieldName: 'value', nodePath: 'positive_prompt', value: `Session prompt ${overrides.id}` }],
  resultImageNames: [],
  sessionId: `session-${overrides.id}`,
  status: 'completed',
  updatedAt: '2026-01-01T00:00:00Z',
  userId: 'user-1',
  ...overrides,
});

let host: HTMLDivElement;
let root: Root;
let projectId: string;
const galleryRecallAll = vi.fn();
const NOT_KEPT =
  "This browser did not keep this item's settings, so only a prompt or seed the server recorded can be recalled.";
let promptBefore: unknown;

const generateValues = () => {
  const snapshot = mocks.store.queries.getSnapshot();
  return getResolvedProjectWidgetValues(snapshot.activeProject, 'generate', snapshot.scopedDefaults);
};

beforeEach(() => {
  vi.clearAllMocks();
  accountLifecycle.activate(`queue-item-recall-${crypto.randomUUID()}`);
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  projectId = project.id;
  project.queue = {
    items: [
      localRun('run-done', 'A recalled castle'),
      localRun('run-failed', 'A failed lighthouse'),
      localRun('run-pending', 'A pending harbour'),
      localWorkflowRun('run-workflow'),
    ],
  };
  mocks.store = createWorkbenchStore(state);
  mocks.store.commands.queue.setConnectionStatus({ status: 'connected' });
  mocks.store.commands.generation.setSettings({ ...getDefaultGenerateSettings(main), model: main }, projectId);
  promptBefore = generateValues().positivePrompt;
  mocks.extensions = createExtensionRegistry();
  mocks.extensions.commands.register({ handler: galleryRecallAll, id: 'viewer.recallAll', title: 'Recall all' });
  host = document.createElement('div');
  host.style.cssText = 'width:360px;height:700px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

const renderQueue = async () => {
  const pending = backendItem({
    id: 4,
    origin: buildQueueItemOrigin('run-pending', projectId),
    status: 'pending',
  });
  const client = new QueryClient();
  client.setQueryData(['queue'], {
    current: null,
    items: [
      pending,
      backendItem({ completedAt: '2026-01-01T00:01:00Z', id: 1, origin: buildQueueItemOrigin('run-done', projectId) }),
      backendItem({
        errorMessage: 'Out of memory',
        id: 2,
        origin: buildQueueItemOrigin('run-failed', projectId),
        status: 'failed',
      }),
      // Queued from another browser: this one kept no snapshot of it.
      backendItem({ id: 3, origin: buildQueueItemOrigin('elsewhere', 'other-project') }),
      // A workflow's text field, which type-and-order matching of session values would read as a prompt.
      backendItem({
        fieldValues: [{ fieldName: 'value', nodePath: 'caption_node', value: 'A workflow caption' }],
        id: 5,
        origin: buildQueueItemOrigin('run-workflow', projectId),
      }),
    ],
    next: pending,
    scope: {},
    status: {
      processor: { isProcessing: false, isStarted: true },
      queue: {
        canceled: 0,
        completed: 3,
        failed: 1,
        inProgress: 0,
        pending: 1,
        queueId: 'default',
        total: 5,
        waiting: 0,
      },
    },
  });
  const { view: QueueWidgetView } = await loadQueueWidgetImplementation();
  const source = { instanceId: 'queue', projectId, region: 'right' as const, typeId: 'queue' };
  const runtime = {
    commands: {
      register: (command: { handler: () => unknown; id: string; title: string }) =>
        mocks.extensions.commands.register({ ...command, source }),
    },
  };

  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={client}>
            <WorkbenchFocusProvider>
              <WorkbenchHotkeyRuntime />
              <QueueUiAdapterProvider>
                <div
                  data-hotkey-widget-instance-id="queue"
                  data-hotkey-widget-region="right"
                  data-hotkey-widget-type-id="queue"
                  style={{ height: 600 }}
                >
                  <QueueWidgetView region="right" runtime={runtime} />
                </div>
              </QueueUiAdapterProvider>
              <div
                data-hotkey-widget-instance-id="gallery"
                data-hotkey-widget-region="right"
                data-hotkey-widget-type-id="gallery"
              >
                <button type="button">Gallery tile</button>
              </div>
            </WorkbenchFocusProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const row = (itemId: number) => host.querySelector<HTMLElement>(`[data-queue-recall-item="${itemId}"]`)!;
const rowButton = (itemId: number) => row(itemId).querySelector<HTMLButtonElement>('[data-list-primary]')!;
const openRowMenu = async (itemId: number) => {
  await act(() =>
    rowButton(itemId).dispatchEvent(
      new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 40, clientY: 40 })
    )
  );
  await vi.waitFor(() => expect(menuItem('Recall all')).not.toBeNull(), { timeout: 5_000 });
};
/** By its label line: an unavailable item also holds its reason's hidden text. */
const menuItem = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
    (candidate) => candidate.querySelector('p')?.textContent === label
  ) ?? null;
const description = (element: HTMLElement) =>
  (element.getAttribute('aria-describedby') ?? '')
    .split(' ')
    .map((id) => document.getElementById(id)?.textContent ?? '')
    .join(' ')
    .trim();
/** Presses A on the target and returns the press, whose `defaultPrevented` says whether a shortcut claimed it. */
const pressA = async (target: HTMLElement): Promise<KeyboardEvent> => {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code: 'KeyA', key: 'a' });
  await act(async () => {
    target.dispatchEvent(event);
    await Promise.resolve();
  });
  return event;
};

describe('Queue entry recall', () => {
  it("recalls every setting of a finished entry from its row's context menu, and says so", async () => {
    await renderQueue();
    await openRowMenu(1);

    expect(menuItem('Recall all')?.getAttribute('aria-disabled')).not.toBe('true');
    await act(() => page.getByRole('menuitem', { exact: true, name: 'Recall all' }).click());

    await vi.waitFor(() => expect(generateValues()).toMatchObject({ positivePrompt: 'A recalled castle', steps: 33 }));
    expect(mocks.openWidget).toHaveBeenCalledWith('generate', { preferredRegions: ['left'] });
    expect(mocks.notify.success).toHaveBeenCalledWith(
      'Recalled image metadata',
      'Loaded these settings into Generate.'
    );
  });

  it('keeps the recall of an entry queued from another browser listed, unavailable and saying why', async () => {
    await renderQueue();
    // Recalling all anyway, with A, explains and writes nothing: the notice ends a recall that does not run.
    rowButton(3).focus();
    await pressA(rowButton(3));
    await vi.waitFor(() => expect(mocks.notify.info).toHaveBeenCalledWith('Nothing recalled', NOT_KEPT));
    expect(mocks.notify.success).not.toHaveBeenCalled();
    expect(generateValues().positivePrompt).toBe(promptBefore);

    await openRowMenu(3);

    // Unavailable while this browser's durable snapshots are searched, then with the reason once none is found.
    await vi.waitFor(() => expect(description(menuItem('Recall all')!)).toBe(NOT_KEPT));
    expect(menuItem('Recall all')!.getAttribute('aria-disabled')).toBe('true');
    // The server recorded the prompt it ran, and the Generate panel's model can take it.
    expect(menuItem('Use prompt')?.getAttribute('aria-disabled')).not.toBe('true');
  });

  it("offers nothing from a workflow run, whose session values are not Generate's", async () => {
    await renderQueue();
    rowButton(5).focus();
    await pressA(rowButton(5));
    await vi.waitFor(() =>
      expect(mocks.notify.info).toHaveBeenCalledWith('Nothing recalled', 'Nothing recallable for this item.')
    );
    expect(generateValues().positivePrompt).toBe(promptBefore);

    await openRowMenu(5);
    await vi.waitFor(() => expect(description(menuItem('Use prompt')!)).toBe('Nothing recallable for this item.'));
    expect(menuItem('Use prompt')!.getAttribute('aria-disabled')).toBe('true');
    expect(description(menuItem('Recall all')!)).toBe('Nothing recallable for this item.');
  });

  it('recalls a failed entry from its expanded details', async () => {
    await renderQueue();
    await act(() => rowButton(2).click());

    const detailButton = () =>
      [...row(2).querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === 'Recall all');
    await vi.waitFor(() => expect(detailButton()?.getAttribute('aria-disabled')).not.toBe('true'), { timeout: 5_000 });
    await vi.waitFor(() => expect(detailButton()?.disabled).toBe(false));
    await act(() => detailButton()!.click());

    await vi.waitFor(() => expect(generateValues()).toMatchObject({ positivePrompt: 'A failed lighthouse' }));
    expect(mocks.notify.success).toHaveBeenCalledWith(
      'Recalled image metadata',
      'Loaded these settings into Generate.'
    );
  });

  it('recalls all from the focused row on A, a pending one included, leaving A to the Gallery elsewhere', async () => {
    await renderQueue();
    const pendingRow = rowButton(4);

    // A queue control that is no row keeps the key.
    const filterTab = host.querySelector<HTMLElement>('[role="tab"]')!;
    filterTab.focus();
    expect((await pressA(filterTab)).defaultPrevented).toBe(false);

    pendingRow.focus();
    expect((await pressA(pendingRow)).defaultPrevented).toBe(true);

    await vi.waitFor(() => expect(generateValues()).toMatchObject({ positivePrompt: 'A pending harbour' }));
    expect(mocks.notify.success).toHaveBeenCalledOnce();
    expect(galleryRecallAll).not.toHaveBeenCalled();

    const galleryTile = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'Gallery tile'
    )!;
    galleryTile.focus();
    await pressA(galleryTile);

    expect(galleryRecallAll).toHaveBeenCalledOnce();
    expect(mocks.notify.success).toHaveBeenCalledOnce();
  });
});
