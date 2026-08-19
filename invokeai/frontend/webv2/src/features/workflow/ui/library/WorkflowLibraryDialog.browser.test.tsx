import type { StarterModel } from '@features/models';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type { InvocationTemplate, InvocationTemplatesSnapshot, ProjectGraphState } from '@features/workflow/core/types';
import type {
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryEntry,
  WorkflowLibraryEntryEnrichment,
} from '@features/workflow/data/libraryBrowseStore';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { buildInvocationNode, createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
// A plain static import of the (mocked) module the dialog `lazy()`-loads, so
// its dynamic `import()` resolves against an already-loaded module record
// instead of paying a first-time compile cost mid-test — that cost is what
// made a cold run of the graph-preview wiring tests below flaky.
import '@features/workflow/ui/graph-preview/GraphPreviewDialog';

import { WorkflowLibraryDialog } from './WorkflowLibraryDialog';

// Task 8's graph preview wiring needs invocation templates loaded to compile
// a library entry's document (`buildLibraryGraphPreviewSource`); this suite
// never boots the real templates fetch, so it stubs the reactive snapshot
// hook with one template — matching the node type `PREVIEW_DOCUMENT` (below)
// uses — the same way `GraphPreviewDialog.browser.test.tsx` does.
const PREVIEW_NODE_TEMPLATE: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputs: {},
  outputType: 'integer_output',
  tags: [],
  title: 'integer',
  type: 'integer',
  useCache: true,
  version: '1.0.0',
};

const TEMPLATES_SNAPSHOT: InvocationTemplatesSnapshot = {
  error: null,
  status: 'loaded',
  templates: { integer: PREVIEW_NODE_TEMPLATE },
};

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => TEMPLATES_SNAPSHOT,
}));

// xyflow stays out of this shell test — the lazy-mounted `GraphPreviewDialog`
// (Task 8) is replaced with a stub that surfaces exactly what the wiring is
// responsible for: which graph got compiled, `hideInvoke`, and the close path.
vi.mock('@features/workflow/ui/graph-preview/GraphPreviewDialog', () => ({
  GraphPreviewDialog: ({
    graphId,
    hideInvoke,
    source,
    sourceLabel,
    onOpenChange,
  }: {
    graphId: string;
    hideInvoke?: boolean;
    isOpen: boolean;
    source: { graph: { nodes: { id: string; type: string }[] } | null };
    sourceLabel: string;
    onOpenChange: (isOpen: boolean) => void;
  }) => (
    <div
      data-graph-id={graphId}
      data-hide-invoke={String(hideInvoke)}
      data-preview-dialog
      data-source-label={sourceLabel}
    >
      {(source.graph?.nodes ?? []).map((node) => node.type).join(',')}
      <button onClick={() => onOpenChange(false)}>Close preview</button>
    </div>
  ),
}));

// The browse store is Task 5's; this suite owns the *dialog*, so the store is
// replaced by a real external store the test drives directly plus spies for
// its four commands. Using a real store (not a stub hook) keeps the dialog's
// subscription, selector equality, and re-render path under test.
const browse = vi.hoisted(() => ({
  ensureWorkflowLibraryBrowseLoaded: vi.fn(() => Promise.resolve()),
  // Assigned by the module factory below, which owns the store instance.
  getSnapshot: null as null | (() => WorkflowLibraryBrowseSnapshot),
  loadNextWorkflowLibraryPage: vi.fn(),
  setSnapshot: null as null | ((snapshot: WorkflowLibraryBrowseSnapshot) => void),
  setWorkflowLibraryBrowseFilter: vi.fn(),
}));

vi.mock('@features/workflow/data/libraryBrowseStore', async () => {
  const { createExternalStore } = await import('@platform/state/externalStore');
  const store = createExternalStore<WorkflowLibraryBrowseSnapshot>({
    entries: [],
    error: null,
    filter: { category: 'user', search: '', tag: null },
    page: 0,
    pages: 0,
    status: 'idle',
    tagCounts: [],
    total: 0,
    userTotal: null,
  });

  browse.getSnapshot = store.getSnapshot;
  browse.setSnapshot = store.setSnapshot;

  return {
    ensureWorkflowLibraryBrowseLoaded: browse.ensureWorkflowLibraryBrowseLoaded,
    getWorkflowLibraryBrowseSnapshot: store.getSnapshot,
    loadNextWorkflowLibraryPage: browse.loadNextWorkflowLibraryPage,
    setWorkflowLibraryBrowseFilter: browse.setWorkflowLibraryBrowseFilter,
    useWorkflowLibraryBrowseSelector: store.useSelector,
  };
});

// The load sequence (fetch → parse → replace → toasts) has its own behavior
// contract; this suite only asserts the dialog *invokes* it and reflects its
// phase in the busy overlay.
const loader = vi.hoisted(() => ({
  load: vi.fn(() => Promise.resolve()),
  phase: { current: 'idle' as 'applying' | 'fetching' | 'idle' },
}));

vi.mock('./useLoadLibraryWorkflow', () => ({
  useLoadLibraryWorkflow: () => ({ load: loader.load, loadPhase: loader.phase.current }),
}));

// The detail panel resolves every entry's requirements against the model
// stores to feed the cards' missing-model badges. Those stores are the models
// feature's; here they are fixed data so the badge under test comes from the
// dialog's own wiring, not a live catalog.
const FLUX_STARTER: StarterModel = {
  base: 'flux',
  description: 'FLUX.1 dev',
  is_installed: false,
  name: 'FLUX.1 dev',
  source: 'https://models.test/flux-dev',
  type: 'main',
};

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  ensureStartersLoaded: vi.fn(),
  useActiveInstallSources: () => new Set<string>(),
  useInstallActions: () => ({ install: vi.fn(), installMany: vi.fn(), pendingSources: new Set() }),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: [] }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ response: { starter_models: [FLUX_STARTER] } }),
}));

// The real i18n client fetches en.json over HTTP, which this browser test
// never boots. Stub `t` with the English strings this dialog renders (plus
// the `_one`/`_other` plural forms) so assertions check real copy.
const TRANSLATIONS: Record<string, string> = {
  'common.close': 'Close',
  'workflowLibrary.allTag': 'All',
  'workflowLibrary.applying': 'Applying workflow…',
  'workflowLibrary.browse': 'Browse',
  'workflowLibrary.delete': 'Delete',
  'workflowLibrary.downloadJson': 'Download JSON',
  'workflowLibrary.duplicate': 'Duplicate',
  'workflowLibrary.empty': 'No workflows match these filters.',
  'workflowLibrary.fetching': 'Fetching workflow…',
  'workflowLibrary.forkIntoProject': 'Fork into new project',
  'workflowLibrary.installModels_one': 'Install 1 model',
  'workflowLibrary.installModels_other': 'Install {{count}} models',
  'workflowLibrary.loading': 'Loading workflows…',
  'workflowLibrary.loadingMore': 'Loading more…',
  'workflowLibrary.moreActions': 'More actions',
  'workflowLibrary.nodeCount_one': '{{count}} node',
  'workflowLibrary.nodeCount_other': '{{count}} nodes',
  'workflowLibrary.notRunYet': 'Not run yet',
  'workflowLibrary.open': 'Open',
  'workflowLibrary.previewGraph': 'Preview graph',
  'workflowLibrary.requirementInstallable': 'Not installed',
  'workflowLibrary.requires': 'Requires',
  'workflowLibrary.sampleOutput': 'Sample output',
  'workflowLibrary.searchPlaceholder': 'Search names, tags, or descriptions',
  'workflowLibrary.title': 'Workflows',
  'workflowLibrary.untitled': 'Untitled Workflow',
  'workflowLibrary.yours': 'Yours',
};

const interpolate = (template: string, options?: Record<string, unknown>): string =>
  options ? template.replaceAll(/\{\{(\w+)\}\}/g, (_match, key: string) => String(options[key] ?? '')) : template;

const translate = (key: string, options?: Record<string, unknown>): string => {
  const count = options?.count;
  const plural = typeof count === 'number' ? TRANSLATIONS[`${key}_${count === 1 ? 'one' : 'other'}`] : undefined;

  return interpolate(plural ?? TRANSLATIONS[key] ?? key, options);
};

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: translate }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const EMPTY_DOCUMENT = createProjectGraph('library-fixture');

// One `integer` node — matches `PREVIEW_NODE_TEMPLATE` above, so this is the
// document the graph-preview wiring tests compile.
const PREVIEW_DOCUMENT: ProjectGraphState = projectGraphReducer(createProjectGraph('preview-fixture'), {
  node: buildInvocationNode(PREVIEW_NODE_TEMPLATE, { x: 0, y: 0 }),
  type: 'addNode',
});

const NO_REQUIREMENTS: readonly WorkflowModelRequirement[] = [];

const readyEnrichment = (
  nodeCount: number,
  primaryBase: string | null,
  requirements: readonly WorkflowModelRequirement[] = NO_REQUIREMENTS,
  document: ProjectGraphState = EMPTY_DOCUMENT
): WorkflowLibraryEntryEnrichment => ({
  document,
  nodeCount,
  requirements: { primaryBase, requirements },
  status: 'ready',
});

const entry = (
  workflowId: string,
  name: string,
  enrichment: WorkflowLibraryEntryEnrichment,
  extras: { tags?: readonly string[]; thumbnailUrl?: string } = {}
): WorkflowLibraryEntry => ({
  enrichment,
  item: {
    category: 'user',
    description: `${name} description`,
    name,
    thumbnail_url: extras.thumbnailUrl ?? null,
    workflow_id: workflowId,
  },
  tags: extras.tags ?? [],
});

const PORTRAIT = entry('wf-portrait', 'Portrait Studio', readyEnrichment(12, 'sdxl'), {
  tags: ['portrait'],
  thumbnailUrl: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
});
// Its one requirement has no installed match but a starter that can fetch it,
// so the grid should badge it as one model to install.
const LANDSCAPE = entry(
  'wf-landscape',
  'Landscape Pass',
  readyEnrichment(7, 'flux', [{ base: 'flux', kind: 'slot', label: 'FLUX checkpoint', modelType: 'main' }]),
  { tags: ['landscape'] }
);
const SKETCH = entry('wf-sketch', 'Sketch To Render', { status: 'pending' });
const UPSCALE = entry(
  'wf-upscale',
  'Upscale Chain',
  { message: 'unreadable', status: 'error' },
  {
    tags: ['upscale'],
  }
);
// The graph-preview wiring's own fixture: a `'ready'` entry whose document
// actually has a node, so the mocked preview dialog has a compiled graph to
// show instead of an empty one.
const PREVIEW_FIXTURE = entry(
  'wf-preview-fixture',
  'Preview Fixture',
  readyEnrichment(1, null, NO_REQUIREMENTS, PREVIEW_DOCUMENT)
);

const LOADED_SNAPSHOT: WorkflowLibraryBrowseSnapshot = {
  entries: [PORTRAIT, LANDSCAPE, SKETCH, UPSCALE],
  error: null,
  filter: { category: 'user', search: '', tag: null },
  page: 0,
  pages: 2,
  status: 'loaded',
  tagCounts: [
    { count: 2, tag: 'portrait' },
    { count: 1, tag: 'upscale' },
  ],
  total: 24,
  userTotal: 4,
};

const withSnapshot = (patch: Partial<WorkflowLibraryBrowseSnapshot>): WorkflowLibraryBrowseSnapshot => ({
  ...LOADED_SNAPSHOT,
  ...patch,
});

// The detail panel's ports. The dialog only has to compose them; their own
// suites cover what they do.
const UI_ADAPTER = {
  notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
} as unknown as WorkflowUiAdapter;
const GRAPH_PREVIEW = { openDocumentInNewProject: vi.fn() } as unknown as WorkflowGraphPreviewPort;

describe('WorkflowLibraryDialog', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;

  /**
   * Chakra's `SegmentGroup` tracks its indicator rect from an observer that
   * commits React state a task after the commit that armed it. Under
   * `StrictMode`'s mount/unmount/remount that lands in the gap *between* two
   * `act` calls, where React's act queue is null — which is what raised the
   * "update ... was not wrapped in act(...)" warnings. Awaiting this **inside**
   * the same `act` scope as the render keeps the queue open across that gap.
   */
  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  const renderDialog = async (isOpen = true) => {
    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={UI_ADAPTER}>
              <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
                <WorkflowLibraryDialog isOpen={isOpen} onOpenChange={onOpenChange} />
              </WorkflowGraphPreviewProvider>
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const openWith = async (snapshot: WorkflowLibraryBrowseSnapshot) => {
    await act(() => browse.setSnapshot?.(snapshot));
    await renderDialog();
  };

  /** Drains the microtask + timer queue the store probe and debounce settle on. */
  const wait = (ms: number) =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, ms);
      });
    });

  /**
   * The lazy-loaded preview dialog (Task 8) resolves its `import()` on its
   * own schedule — a fixed `wait` is either too short (flaky) or padded
   * (slow), and letting it resolve outside `act` is what raises "a suspended
   * resource finished loading" warnings. Polling inside `act` keeps every
   * check, and the eventual resolution, in the same act scope.
   */
  const waitForPreviewDialog = () =>
    act(async () => {
      await vi.waitFor(
        () => {
          expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();
        },
        // Padded past `vi.waitFor`'s default 1000ms: the static side-effect
        // import above keeps the module warm, but the dynamic `import()`
        // still costs a real (if now small) round trip through Vite's module
        // graph before `Suspense` re-renders.
        { timeout: 2000 }
      );
    });

  const cards = () => [...document.querySelectorAll<HTMLElement>('[data-workflow-card]')];
  const card = (workflowId: string) => document.querySelector<HTMLElement>(`[data-workflow-card="${workflowId}"]`);
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '') === text);

  const clickText = async (text: string) => {
    const button = buttonWithText(text);
    expect(button).not.toBeUndefined();

    await act(() => button?.click());
  };

  const clickSegment = async (value: string) => {
    const label = document.querySelector<HTMLInputElement>(`input[value="${value}"]`)?.closest('label');
    expect(label).not.toBeNull();

    await act(async () => {
      label?.click();
      await settleFrame();
    });
  };

  const gridViewport = () => document.querySelector<HTMLElement>('[role="region"][aria-label="Workflows"]');

  /**
   * Real layout in a `Dialog` portal gives no deterministic scroll geometry,
   * so the viewport's metrics are defined directly — the assertion is about
   * the dialog's near-bottom threshold, not the browser's layout engine.
   */
  const scrollTo = async (scrollTop: number) => {
    const viewport = gridViewport();
    expect(viewport).not.toBeNull();

    for (const [property, value] of [
      ['clientHeight', 400],
      ['scrollHeight', 4000],
      ['scrollTop', scrollTop],
    ] as const) {
      Object.defineProperty(viewport, property, { configurable: true, value, writable: true });
    }

    await act(async () => {
      viewport?.dispatchEvent(new Event('scroll'));
      // The ScrollArea machine reacts to the scroll a task later; see
      // `settleFrame` for why that has to stay inside this `act` scope.
      await settleFrame();
    });
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    browse.ensureWorkflowLibraryBrowseLoaded.mockClear();
    browse.loadNextWorkflowLibraryPage.mockClear();
    browse.setWorkflowLibraryBrowseFilter.mockClear();
    loader.load.mockClear();
    loader.phase.current = 'idle';
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('renders one card per entry with its name and enriched node count', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(cards()).toHaveLength(4);

    const text = document.body.textContent ?? '';

    expect(text).toContain('Portrait Studio');
    expect(text).toContain('12 nodes');
    expect(text).toContain('SDXL');
    expect(text).toContain('Landscape Pass');
    expect(text).toContain('7 nodes');
    expect(text).toContain('FLUX');
  });

  it('shows a quiet placeholder while enrichment is pending and omits the badges when it failed', async () => {
    await openWith(LOADED_SNAPSHOT);

    const pending = card('wf-sketch');
    expect(pending?.textContent).toContain('Sketch To Render');
    expect(pending?.textContent).not.toContain('nodes');
    expect(pending?.querySelector('[data-enrichment-placeholder]')).not.toBeNull();

    // A failed enrichment is a quiet omission, never an error-styled card.
    const failed = card('wf-upscale');
    expect(failed?.textContent).toContain('Upscale Chain');
    expect(failed?.textContent).not.toContain('nodes');
    expect(failed?.querySelector('[data-enrichment-placeholder]')).toBeNull();
  });

  it('renders the thumbnail when present and the not-run-yet placeholder when absent', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-portrait')?.querySelector('img')).not.toBeNull();
    expect(card('wf-landscape')?.querySelector('img')).toBeNull();
    expect(card('wf-landscape')?.textContent).toContain('Not run yet');
  });

  it('kicks the first load when it opens and leaves the store alone while closed', async () => {
    await act(() => browse.setSnapshot?.(LOADED_SNAPSHOT));
    await renderDialog(false);

    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();

    await renderDialog(true);

    // Times are not asserted: StrictMode double-invokes mount effects, and the
    // store's own single-flight is what makes that a no-op.
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalled();
  });

  it('switches to the bundled defaults once when the account has no workflows of its own', async () => {
    await openWith(withSnapshot({ userTotal: 0 }));
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ category: 'default', tag: null });
  });

  it('leaves the category alone when the account already has workflows', async () => {
    await openWith(LOADED_SNAPSHOT);
    await wait(0);

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalled();
  });

  it('resets the tag when the category segment changes', async () => {
    await openWith(withSnapshot({ filter: { category: 'user', search: '', tag: 'portrait' } }));

    await clickSegment('default');

    // The store applies filter patches literally, so clearing the tag on a
    // category switch is the dialog's job.
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ category: 'default', tag: null });
  });

  it('selects and clears a tag from the chip row', async () => {
    await openWith(LOADED_SNAPSHOT);

    await clickText('portrait2');

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ tag: 'portrait' });

    await clickText('All');

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ tag: null });
  });

  it('pushes the search term to the store once, after the debounce', async () => {
    await openWith(LOADED_SNAPSHOT);

    const input = document.querySelector<HTMLInputElement>('input[type="search"]');
    expect(input).not.toBeNull();

    // React tracks the input's value through its own setter, so assigning
    // `.value` directly would be swallowed as a no-op change.
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

    for (const value of ['p', 'ph', 'pho', 'phot', 'photo']) {
      await act(async () => {
        setValue?.call(input, value);
        input?.dispatchEvent(new Event('input', { bubbles: true }));
        await Promise.resolve();
      });
    }

    expect(browse.setWorkflowLibraryBrowseFilter).not.toHaveBeenCalled();

    await wait(400);

    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledTimes(1);
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ search: 'photo' });
  });

  it('requests the next page only once the grid is scrolled near the bottom', async () => {
    await openWith(LOADED_SNAPSHOT);

    await scrollTo(100);

    expect(browse.loadNextWorkflowLibraryPage).not.toHaveBeenCalled();

    await scrollTo(3200);

    expect(browse.loadNextWorkflowLibraryPage).toHaveBeenCalled();
  });

  it('auto-selects the first entry and moves the selection when it disappears', async () => {
    await openWith(LOADED_SNAPSHOT);

    expect(card('wf-portrait')?.getAttribute('aria-pressed')).toBe('true');
    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('false');

    await act(() => card('wf-landscape')?.click());

    expect(card('wf-landscape')?.getAttribute('aria-pressed')).toBe('true');

    // A filter change drops the selected row; selection falls back to the head
    // of the new list rather than pointing at nothing.
    await act(() => browse.setSnapshot?.(withSnapshot({ entries: [SKETCH, UPSCALE] })));

    expect(card('wf-sketch')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('opens a workflow on double click', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => {
      card('wf-landscape')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });

    expect(loader.load).toHaveBeenCalledTimes(1);
    expect(loader.load).toHaveBeenCalledWith(LANDSCAPE.item);
  });

  it('dedupes entries that share a workflow id', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, LANDSCAPE, { ...PORTRAIT }] }));

    expect(cards()).toHaveLength(2);
    expect(document.querySelectorAll('[data-workflow-card="wf-portrait"]')).toHaveLength(1);
  });

  it('keeps the loaded grid and reports a failed load-more inline', async () => {
    await openWith(withSnapshot({ error: 'Failed to load more workflows.', status: 'error' }));

    expect(cards()).toHaveLength(4);
    expect(document.body.textContent ?? '').toContain('Failed to load more workflows.');
    expect(document.body.textContent ?? '').not.toContain('No workflows match these filters.');
  });

  it('shows the loading copy on the very first paint, before the store leaves idle', async () => {
    // The pristine store: `ensureWorkflowLibraryBrowseLoaded` only flips the
    // status to 'loading' from the open-time effect, so the first paint of a
    // freshly-booted session renders against 'idle'. That must never read as
    // "nothing matched".
    await openWith({
      entries: [],
      error: null,
      filter: { category: 'user', search: '', tag: null },
      page: 0,
      pages: 0,
      status: 'idle',
      tagCounts: [],
      total: 0,
      userTotal: null,
    });

    const text = document.body.textContent ?? '';

    expect(text).toContain('Loading workflows…');
    expect(text).not.toContain('No workflows match these filters.');
  });

  it('shows the empty state only when nothing loaded', async () => {
    await openWith(withSnapshot({ entries: [], status: 'loaded', total: 0 }));

    expect(cards()).toHaveLength(0);
    expect(document.body.textContent ?? '').toContain('No workflows match these filters.');
  });

  it('shows the selected workflow in the rail and follows the selection', async () => {
    await openWith(LOADED_SNAPSHOT);

    const detail = () => document.querySelector<HTMLElement>('[data-workflow-detail]');

    expect(detail()?.dataset.workflowDetail).toBe('wf-portrait');

    await act(() => card('wf-landscape')?.click());

    expect(detail()?.dataset.workflowDetail).toBe('wf-landscape');
    expect(detail()?.textContent).toContain('Landscape Pass');
  });

  it('opens the selected workflow from the rail, the keyboard-reachable path', async () => {
    await openWith(LOADED_SNAPSHOT);

    // Portrait needs nothing installed, so the rail's primary action is Open.
    await clickText('Open');

    expect(loader.load).toHaveBeenCalledWith(PORTRAIT.item);
  });

  it('badges the cards with the models their workflows still need', async () => {
    await openWith(LOADED_SNAPSHOT);

    // Resolved from the same model data the rail uses: the FLUX slot has no
    // installed match but a starter that can fetch it.
    expect(card('wf-landscape')?.textContent).toContain('Install 1 model');
    expect(card('wf-portrait')?.textContent).not.toContain('Install');
  });

  it('records the workflow the rail asked to preview', async () => {
    await openWith(LOADED_SNAPSHOT);

    await clickText('Preview graph');

    // Task 8 mounts the preview dialog from this pending selection.
    expect(document.querySelector('[data-pending-preview="wf-portrait"]')).not.toBeNull();
  });

  it("mounts the lazy preview dialog with the entry's compiled graph and hides Invoke", async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    const preview = document.querySelector('[data-preview-dialog]');
    expect(preview).not.toBeNull();
    expect(preview?.getAttribute('data-graph-id')).toBe('wf-preview-fixture');
    expect(preview?.getAttribute('data-hide-invoke')).toBe('true');
    expect(preview?.getAttribute('data-source-label')).toBe('Preview Fixture');
    // The document's one `integer` node made it through compilation.
    expect(preview?.textContent).toContain('integer');
  });

  it('disables the Preview action for an entry whose enrichment is not ready', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => card('wf-sketch')?.click());

    const button = buttonWithText('Preview graph') as HTMLButtonElement | undefined;
    expect(button?.disabled).toBe(true);
  });

  it('resets the pending preview when the preview dialog itself closes', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();

    await clickText('Close preview');

    expect(document.querySelector('[data-preview-dialog]')).toBeNull();
    expect(document.querySelector('[data-pending-preview]')).toBeNull();
  });

  it('does not resurrect the preview after the library dialog closes and reopens', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();

    // The dialog's own Close control is the only path that ever flips `isOpen`
    // to `false` in the real app (`WorkflowWidgetChrome` only ever sets it
    // back to `true`) — clicking it exercises the same reset the parent
    // relies on. The isOpen round trip below then confirms the closed render
    // drops the mount and the reopened one does not bring it back.
    const closeButton = document.querySelector<HTMLButtonElement>('button[aria-label="Close"]');
    expect(closeButton).not.toBeNull();
    // Chakra's `Dialog` commits its close transition a task after the click
    // (the same class of gap `settleFrame` exists for, above) — settle inside
    // this `act` scope rather than a bare click.
    await act(async () => {
      closeButton?.click();
      await settleFrame();
    });

    await renderDialog(false);
    expect(document.querySelector('[data-preview-dialog]')).toBeNull();

    await renderDialog(true);
    expect(document.querySelector('[data-preview-dialog]')).toBeNull();
  });

  it('shows the busy overlay while a workflow is being applied', async () => {
    loader.phase.current = 'applying';

    await openWith(LOADED_SNAPSHOT);

    const status = document.querySelector('[role="status"]');
    expect(status?.textContent).toContain('Applying workflow…');
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull();
  });
});
