import type { StarterModel } from '@features/models';
import type { WorkflowModelRequirement } from '@features/workflow/core/modelRequirements';
import type {
  InvocationTemplate,
  InvocationTemplatesSnapshot,
  ProjectGraphState,
  ProjectWorkflowEntry,
} from '@features/workflow/core/types';
import type {
  WorkflowLibraryBrowseSnapshot,
  WorkflowLibraryEntry,
  WorkflowLibraryEntryEnrichment,
} from '@features/workflow/data/libraryBrowseStore';
import type {
  WorkflowGraphPreviewPort,
  WorkflowReadPort,
  WorkflowUiAdapter,
} from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import {
  openWorkflowLibraryAtProjectWorkflow,
  setWorkflowLibraryTab,
  workflowUiStore,
} from '@features/workflow/ui/workflowUiStore';
import { buildInvocationNode, createProjectGraph, projectGraphReducer } from '@features/workflow/utility';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
// Warm the mocked lazy module to avoid first-compile timing variability in preview wiring tests.
import '@features/workflow/ui/graph-preview/GraphPreviewDialog';

import { WorkflowLibraryDialog } from './WorkflowLibraryDialog';

// Provide the fixture node's reactive template snapshot so preview compilation can run without backend schema
// loading.
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

// Stub preview rendering while exposing compiled graph, hideInvoke, and close behavior owned by dialog wiring.
vi.mock('@features/workflow/ui/graph-preview/GraphPreviewDialog', () => ({
  GraphPreviewDialog: ({
    graphId,
    hideInvoke,
    isOpen,
    source,
    sourceLabel,
    onExitComplete,
    onOpenChange,
  }: {
    graphId: string;
    hideInvoke?: boolean;
    isOpen: boolean;
    source: { graph: { nodes: { id: string; type: string }[] } | null };
    sourceLabel: string;
    onExitComplete?: () => void;
    onOpenChange: (isOpen: boolean) => void;
  }) => (
    <div
      data-graph-id={graphId}
      data-hide-invoke={String(hideInvoke)}
      data-preview-dialog
      data-preview-open={String(isOpen)}
      data-source-label={sourceLabel}
    >
      {(source.graph?.nodes ?? []).map((node) => node.type).join(',')}
      <button onClick={() => onOpenChange(false)}>Close preview</button>
      {/* Stands in for Ark's presence machine: the real dialog reports the end
          of its close transition, which is what releases the mount. */}
      <button onClick={() => onExitComplete?.()}>Finish preview exit</button>
    </div>
  ),
}));

// Drive a real external browse store to retain subscription and selector behavior while spying on commands.
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

// Test opener invocation and busy-state wiring here; load sequencing has separate coverage.
const loader = vi.hoisted(() => ({
  open: vi.fn((_item: unknown, _mode: 'resume-or-add' | 'add-copy') => Promise.resolve()),
  phase: { current: 'idle' as 'applying' | 'fetching' | 'idle' },
  resume: vi.fn((_workflowId: string) => {}),
}));

vi.mock('./useOpenLibraryWorkflow', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useOpenLibraryWorkflow: () => ({ loadPhase: loader.phase.current, open: loader.open, resume: loader.resume }),
}));

// Use fixed model-store data so missing-model badges test dialog wiring rather than live catalogs.
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

// Provide English/plural strings without booting the HTTP-backed i18n client.
const TRANSLATIONS: Record<string, string> = {
  'common.close': 'Close',
  'workflowLibrary.activeWorkflow': 'Active',
  'workflowLibrary.addAnotherCopy': 'Add another copy',
  'workflowLibrary.addAnotherCopyHint': 'A second, independent copy in this project',
  'workflowLibrary.addWorkflow': 'Add workflow',
  'workflowLibrary.chooseProjectCopy': 'Open which copy?',
  'workflowLibrary.allTag': 'All',
  'workflowLibrary.applying': 'Applying workflow…',
  'workflowLibrary.browse': 'Browse',
  'workflowLibrary.delete': 'Delete',
  'workflowLibrary.downloadJson': 'Download JSON',
  'workflowLibrary.duplicate': 'Duplicate',
  'workflowLibrary.duplicateName': '{{name}} copy',
  'workflowLibrary.empty': 'No workflows match these filters.',
  'workflowLibrary.fetching': 'Fetching workflow…',
  'workflowLibrary.forkIntoProject': 'Fork into new project',
  'workflowLibrary.installModels_one': 'Install 1 model',
  'workflowLibrary.installModels_other': 'Install {{count}} models',
  'workflowLibrary.loading': 'Loading workflows…',
  'workflowLibrary.loadingMore': 'Loading more…',
  'workflowLibrary.moreActions': 'More actions',
  'workflowLibrary.newWorkflow': 'New workflow',
  'workflowLibrary.nodeCount_one': '{{count}} node',
  'workflowLibrary.nodeCount_other': '{{count}} nodes',
  'workflowLibrary.notRunYet': 'Not run yet',
  'workflowLibrary.open': 'Open',
  'workflowLibrary.openHint': 'Adds a copy to this project',
  'workflowLibrary.openProjectCopy': 'Open project copy',
  'workflowLibrary.openProjectCopyHint': 'Switches to the copy this project already has',
  'workflowLibrary.previewGraph': 'Preview graph',
  'workflowLibrary.projectWorkflowCount_one': '{{count}} workflow in this project',
  'workflowLibrary.projectWorkflowCount_other': '{{count}} workflows in this project',
  'workflowLibrary.remove': 'Remove',
  'workflowLibrary.removeConfirmBody':
    'Remove "{{name}}" from this project? Its library template, if any, is not affected.',
  'workflowLibrary.removeConfirmTitle': 'Remove workflow',
  'workflowLibrary.removeWithEllipsis': 'Remove…',
  'workflowLibrary.rename': 'Rename',
  'workflowLibrary.renameTitle': 'Rename workflow',
  'workflowLibrary.renameWithEllipsis': 'Rename…',
  'workflowLibrary.requirementInstallable': 'Not installed',
  'workflowLibrary.requires': 'Requires',
  'workflowLibrary.sampleOutput': 'Sample output',
  'workflowLibrary.saveToLibraryWithEllipsis': 'Save to library…',
  'workflowLibrary.searchPlaceholder': 'Search names, tags, or descriptions',
  'workflowLibrary.sourceBundled': 'From a bundled template (read-only)',
  'workflowLibrary.sourceNone': 'Not linked to a library template',
  'workflowLibrary.sourceRevision': 'Linked to library template at revision {{revision}}',
  'workflowLibrary.thisProject': 'This project',
  'workflowLibrary.thisProjectHint':
    'Workflows saved with this project. Edits autosave here; the library changes only when you save to it.',
  'workflowLibrary.title': 'Workflows',
  'workflowLibrary.untitled': 'Untitled Workflow',
  'workflowLibrary.updateTemplate': 'Update library template',
  'workflowLibrary.workflowName': 'Workflow name',
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
    revision: 1,
    thumbnail_url: extras.thumbnailUrl ?? null,
    workflow_id: workflowId,
  },
  tags: extras.tags ?? [],
});

const PORTRAIT = entry('wf-portrait', 'Portrait Studio', readyEnrichment(12, 'sdxl'), {
  tags: ['portrait'],
  thumbnailUrl: 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==',
});
// One absent requirement with a matching starter should show one model to install.
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
const PREVIEW_FIXTURE = entry(
  'wf-preview-fixture',
  'Preview Fixture',
  readyEnrichment(1, null, NO_REQUIREMENTS, PREVIEW_DOCUMENT)
);

/** ~80 characters with no break opportunity the rail could exploit. */
const LONG_NAME = 'Cinematic Portrait Restoration And Upscaling Pipeline With Refiner And Face Detail';
const LONG_NAMED = entry('wf-long-name', LONG_NAME, readyEnrichment(3, 'sdxl'));

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

const createMutablePort = <Snapshot,>(initialSnapshot: Snapshot) => {
  let snapshot = initialSnapshot;
  const listeners = new Set<() => void>();
  const port: WorkflowReadPort<Snapshot> = {
    getSnapshot: () => snapshot,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  return {
    port,
    setSnapshot: (next: Snapshot) => {
      snapshot = next;
      for (const listener of listeners) {
        listener();
      }
    },
  };
};

const PROJECT_ID = 'project-1';

/** The project's own collection: an unlinked active workflow, a bundled copy, and a copy of a user template. */
const ALPHA: ProjectWorkflowEntry = { document: { ...createProjectGraph('wf-alpha'), name: 'Alpha' } };
const BUNDLED_COPY: ProjectWorkflowEntry = {
  document: { ...createProjectGraph('wf-bundled'), name: 'Bundled copy' },
  source: { libraryWorkflowId: 'default_text_to_image', revision: 1 },
};
const USER_COPY: ProjectWorkflowEntry = {
  document: { ...PREVIEW_DOCUMENT, id: 'wf-user', name: 'User copy' },
  source: { libraryWorkflowId: 'wf-portrait', revision: 4 },
};
const PROJECT_WORKFLOWS: readonly ProjectWorkflowEntry[] = [ALPHA, BUNDLED_COPY, USER_COPY];

const projectSnapshot = (workflows: readonly ProjectWorkflowEntry[], activeWorkflowId = 'wf-alpha') => {
  const activeWorkflow = workflows.find((entry) => entry.document.id === activeWorkflowId) ?? workflows[0]!;

  return {
    activeWorkflow,
    activeWorkflowId: activeWorkflow.document.id,
    galleryValues: {},
    id: PROJECT_ID,
    isWorkflowRunning: false,
    projectGraph: activeWorkflow.document,
    workflowValues: {},
    workflows,
  };
};

const project = createMutablePort(projectSnapshot(PROJECT_WORKFLOWS));
const COMMANDS = {
  addWorkflow: vi.fn(() => 'wf-added'),
  createWorkflow: vi.fn(() => 'wf-new'),
  duplicateWorkflow: vi.fn((_workflowId: string, _copyName: string) => 'wf-copy'),
  editGraph: vi.fn(),
  redo: vi.fn(),
  removeWorkflow: vi.fn(),
  renameWorkflow: vi.fn(),
  selectWorkflow: vi.fn(),
  setWorkflowSource: vi.fn(),
  undo: vi.fn(),
};
const UI_ADAPTER = {
  commands: COMMANDS,
  getProjectGraph: () => project.port.getSnapshot().projectGraph,
  notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  openAddModels: vi.fn(),
  project: project.port,
} as unknown as WorkflowUiAdapter;
const GRAPH_PREVIEW = { openDocumentInNewProject: vi.fn() } as unknown as WorkflowGraphPreviewPort;

describe('WorkflowLibraryDialog', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;

  /** Await deferred SegmentGroup observer updates inside the render act scope, including StrictMode remounts. */
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

  /** Poll lazy preview resolution inside act to avoid timing-dependent waits and suspended-resource warnings. */
  const waitForPreviewDialog = () =>
    act(async () => {
      await vi.waitFor(
        () => {
          expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();
        },
        // Allow time for Vite's dynamic-import round trip even when the module is warm.
        { timeout: 2000 }
      );
    });

  const cards = () => [...document.querySelectorAll<HTMLElement>('[data-workflow-card]')];
  const card = (workflowId: string) => document.querySelector<HTMLElement>(`[data-workflow-card="${workflowId}"]`);
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '') === text);

  /** A pointer press on a tile's menu button: pointerdown (an outside press for an open menu), then the click. */
  const pressTileMenuButton = async (button: HTMLElement | null) => {
    expect(button).not.toBeNull();
    await act(async () => {
      button?.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerType: 'mouse' }));
      await settleFrame();
      button?.click();
      await settleFrame();
      await settleFrame();
    });
  };

  const clickText = async (text: string) => {
    const button = buttonWithText(text);
    expect(button).not.toBeUndefined();

    await act(() => button?.click());
  };

  const clickSegment = async (value: string) => {
    const tab = document.querySelector<HTMLButtonElement>(`[role="tab"][id$="-tab-${value}"]`);
    expect(tab).not.toBeNull();

    await act(async () => {
      tab?.click();
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
      // Settle deferred ScrollArea reactions within the same act scope.
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
    loader.open.mockClear();
    loader.resume.mockClear();
    loader.phase.current = 'idle';
    for (const command of Object.values(COMMANDS)) {
      command.mockClear();
    }
    project.setSnapshot(projectSnapshot(PROJECT_WORKFLOWS));
    // The dialog's tab and selection live in the session store; start every test from the account's clean slate.
    accountLifecycle.invalidate();
    // These are the template tabs' behaviours; the project view has its own suite below.
    setWorkflowLibraryTab('user');
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
    // The segment follows the store's category, so the tab moves with it.
    expect(workflowUiStore.getSnapshot().libraryTab).toBe('default');
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

    await act(() => browse.setSnapshot?.(withSnapshot({ entries: [SKETCH, UPSCALE] })));

    expect(card('wf-sketch')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('opens a workflow on double click', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() => {
      card('wf-landscape')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    });

    expect(loader.open).toHaveBeenCalledTimes(1);
    expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'resume-or-add');
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
    // Initial idle state precedes loading and must not render as no matches.
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

  it("opens a tile's actions from its own menu button, toggles it closed, and hands focus back to the button", async () => {
    await openWith(LOADED_SNAPSHOT);

    const button = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-landscape"]');
    expect(button).not.toBeNull();
    // A sibling of the card button, never nested inside it.
    expect(document.querySelector('[data-workflow-card="wf-landscape"]')?.tagName).toBe('BUTTON');
    expect(document.querySelector('[data-workflow-card="wf-landscape"]')?.contains(button)).toBe(false);

    await act(async () => {
      button?.click();
      await settleFrame();
    });

    // The button selects its tile and opens the same actions the right-click offers, anchored to the button.
    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-landscape');
    const open = () =>
      document.querySelector<HTMLElement>('[data-workflow-context-menu][data-state="open"] [data-menu-item="open"]');
    await vi.waitFor(() => expect(open()).not.toBeNull());
    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull()
    );
    expect(document.activeElement).toBe(button);

    // A second press on the same button closes the menu instead of reopening it elsewhere. The press is a pointer
    // sequence, since the pointerdown lands outside the menu and must be kept from dismissing it first.
    await pressTileMenuButton(button);
    await vi.waitFor(() => expect(open()).not.toBeNull());
    expect(button?.getAttribute('aria-expanded')).toBe('true');
    await pressTileMenuButton(button);
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).toBeNull()
    );
    expect(button?.getAttribute('aria-expanded')).toBe('false');
  });

  it("moves the open menu to another tile's button and hands the expanded state over", async () => {
    await openWith(LOADED_SNAPSHOT);

    const first = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-landscape"]');
    const second = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-portrait"]');
    expect(second).not.toBeNull();

    await pressTileMenuButton(first);
    await vi.waitFor(() => expect(first?.getAttribute('aria-expanded')).toBe('true'));

    await pressTileMenuButton(second);

    await vi.waitFor(() => expect(second?.getAttribute('aria-expanded')).toBe('true'));
    expect(first?.getAttribute('aria-expanded')).toBe('false');
    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-portrait');
    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu][data-state="open"]')).not.toBeNull()
    );
  });

  it('hands focus to the template rail menu it opens', async () => {
    await openWith(LOADED_SNAPSHOT);
    await act(() => card('wf-landscape')?.click());

    const trigger = document.querySelector<HTMLElement>('[data-workflow-detail] [aria-label="More actions"]');
    expect(trigger).not.toBeNull();
    await act(async () => {
      trigger?.click();
      await settleFrame();
    });

    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });
  });

  it('opens the rail actions for a card from a right-click and runs them', async () => {
    await openWith(LOADED_SNAPSHOT);

    await act(() =>
      card('wf-landscape')?.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 120, clientY: 80 })
      )
    );

    // The right-click selects the card, so the rail (which owns the actions) shows it.
    expect(document.querySelector<HTMLElement>('[data-workflow-detail]')?.dataset.workflowDetail).toBe('wf-landscape');

    const open = () => document.querySelector<HTMLElement>('[data-workflow-context-menu] [data-menu-item="open"]');

    await vi.waitFor(() => expect(open()).not.toBeNull());
    await act(() => open()?.click());

    expect(loader.open).toHaveBeenCalledWith(LANDSCAPE.item, 'resume-or-add');
  });

  it('closes the card context menu on Escape and hands focus back to the card', async () => {
    await openWith(LOADED_SNAPSHOT);

    // A keyboard-raised menu reports no pointer position.
    await act(() =>
      card('wf-landscape')?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }))
    );

    const menu = () => document.querySelector<HTMLElement>('[data-workflow-context-menu]');

    // The positioner places the menu a frame after it mounts; anchored inside the card, not at the viewport corner.
    await vi.waitFor(() =>
      expect(menu()?.getBoundingClientRect().left ?? 0).toBeGreaterThan(
        card('wf-landscape')!.getBoundingClientRect().left
      )
    );

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });

    await vi.waitFor(() => expect(menu()).toBeNull());
    await vi.waitFor(() => expect(document.activeElement).toBe(card('wf-landscape')));
  });

  it('opens the selected workflow from the rail, the keyboard-reachable path', async () => {
    await openWith(LOADED_SNAPSHOT);

    // Portrait needs nothing installed, but the project already holds a copy of it: Open resumes that copy.
    await clickText('Open project copy');

    expect(loader.resume).toHaveBeenCalledWith('wf-user');
    expect(loader.open).not.toHaveBeenCalled();

    // Upscale has no copy yet (and nothing to install first), so Open adds one.
    await act(() => card('wf-upscale')?.click());
    await clickText('Open');

    expect(loader.open).toHaveBeenCalledWith(UPSCALE.item, 'resume-or-add');
  });

  it('badges the cards with the models their workflows still need', async () => {
    await openWith(LOADED_SNAPSHOT);

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

  it('keeps the preview mounted through its exit transition, then resets the pending preview', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');

    await clickText('Close preview');

    // Keep the closing preview mounted until its exit animation finishes.
    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('false');
    expect(document.querySelector('[data-pending-preview]')).not.toBeNull();

    await clickText('Finish preview exit');

    expect(document.querySelector('[data-preview-dialog]')).toBeNull();
    expect(document.querySelector('[data-pending-preview]')).toBeNull();
  });

  it('keeps a re-opened preview mounted when the previous exit reports in late', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    await clickText('Close preview');
    // Previewing again mid-transition re-opens the same dialog…
    await clickText('Preview graph');

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');

    // …and the exit report from the *previous* close must not unmount it.
    await clickText('Finish preview exit');

    expect(document.querySelector('[data-preview-dialog]')?.getAttribute('data-preview-open')).toBe('true');
  });

  it('does not resurrect the preview after the library dialog closes and reopens', async () => {
    await openWith(withSnapshot({ entries: [PORTRAIT, PREVIEW_FIXTURE] }));

    await act(() => card('wf-preview-fixture')?.click());
    await clickText('Preview graph');
    await waitForPreviewDialog();

    expect(document.querySelector('[data-preview-dialog]')).not.toBeNull();

    // Use the dialog's real Close path, then reopen through controlled props to verify reset and mount cleanup.
    const closeButton = document.querySelector<HTMLButtonElement>('button[aria-label="Close"]');
    expect(closeButton).not.toBeNull();
    // Settle deferred Dialog close transitions inside act.
    await act(async () => {
      closeButton?.click();
      await settleFrame();
    });

    await renderDialog(false);
    expect(document.querySelector('[data-preview-dialog]')).toBeNull();

    await renderDialog(true);
    expect(document.querySelector('[data-preview-dialog]')).toBeNull();
  });

  it('sits the close control on the header row, not floating over the tag chips', async () => {
    await openWith(LOADED_SNAPSHOT);

    const close = document.querySelector<HTMLButtonElement>('button[aria-label="Close"]');
    const segments = document.querySelector<HTMLElement>('[role="tablist"][aria-label="Workflows"]');
    expect(close).not.toBeNull();
    expect(segments).not.toBeNull();

    // Same row container as the title/search/tab cluster…
    expect(segments?.parentElement?.parentElement?.contains(close as Node)).toBe(true);
    // …and on its baseline rather than the dialog's absolute top corner.
    const closeBox = close?.getBoundingClientRect();
    const segmentBox = segments?.getBoundingClientRect();
    const closeCenter = (closeBox?.top ?? 0) + (closeBox?.height ?? 0) / 2;

    expect(closeCenter).toBeGreaterThanOrEqual(segmentBox?.top ?? 0);
    expect(closeCenter).toBeLessThanOrEqual(segmentBox?.bottom ?? 0);
    // And it comes after the segment control, at the end of the row.
    expect(closeBox?.left ?? 0).toBeGreaterThanOrEqual(segmentBox?.right ?? 0);
  });

  it('wraps a long workflow name in full instead of scrolling the rail sideways', async () => {
    await openWith(withSnapshot({ entries: [LONG_NAMED, PORTRAIT] }));

    const detail = document.querySelector<HTMLElement>('[data-workflow-detail]');
    const body = detail?.parentElement;
    expect(detail?.dataset.workflowDetail).toBe('wf-long-name');
    expect(body).not.toBeNull();

    // 18rem rail plus its 1px borders. Anything wider is the rail growing to
    // its content's min-content width.
    expect(detail?.getBoundingClientRect().width).toBeLessThanOrEqual(18 * 16 + 2);
    expect(body?.scrollWidth).toBeLessThanOrEqual((body?.clientWidth ?? 0) + 1);

    // The rail's own (vertical) scroll area must not scroll horizontally: it
    // renders no horizontal scrollbar, so overflow there just hides content.
    const railViewport = detail?.querySelector<HTMLElement>(`[role="region"][aria-label="${LONG_NAME}"]`);
    expect(railViewport).not.toBeNull();
    expect(railViewport?.scrollWidth).toBeLessThanOrEqual((railViewport?.clientWidth ?? 0) + 1);

    // The detail rail must wrap the full workflow name without clipping.
    const heading = [...(detail?.querySelectorAll('p') ?? [])].find((element) => element.textContent === LONG_NAME);
    expect(heading, 'the full name should be rendered').not.toBeUndefined();
    expect(heading?.scrollWidth).toBeLessThanOrEqual((heading?.clientWidth ?? 0) + 1);
    // Wrapped, not squeezed onto one line.
    expect(heading?.getBoundingClientRect().height ?? 0).toBeGreaterThan(20);
  });

  it('shows the busy overlay while a workflow is being applied', async () => {
    loader.phase.current = 'applying';

    await openWith(LOADED_SNAPSHOT);

    const status = document.querySelector('[role="status"]');
    expect(status?.textContent).toContain('Applying workflow…');
    expect(document.querySelector('[aria-busy="true"]')).not.toBeNull();
  });
});

describe('WorkflowLibraryDialog — This project', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;

  const settleFrame = () =>
    new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });

  const renderDialog = async () => {
    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={UI_ADAPTER}>
              <WorkflowGraphPreviewProvider adapter={GRAPH_PREVIEW}>
                <WorkflowLibraryDialog isOpen onOpenChange={onOpenChange} />
              </WorkflowGraphPreviewProvider>
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      );
      await settleFrame();
    });
  };

  const cards = () => [...document.querySelectorAll<HTMLElement>('[data-workflow-card]')];
  const card = (workflowId: string) => document.querySelector<HTMLElement>(`[data-workflow-card="${workflowId}"]`);
  const rail = () => document.querySelector<HTMLElement>('[data-project-workflow-detail]');
  const buttonWithText = (text: string) =>
    [...document.querySelectorAll('button')].find((candidate) => (candidate.textContent ?? '') === text);
  const alertDialog = () => document.querySelector<HTMLElement>('[role="alertdialog"]');

  const click = async (element: HTMLElement | null | undefined) => {
    expect(element).not.toBeFalsy();

    await act(async () => {
      element?.click();
      await settleFrame();
    });
  };
  const clickText = (text: string) => click(buttonWithText(text));
  const selectCard = (workflowId: string) => click(card(workflowId));

  const menuItem = (value: string) => document.querySelector<HTMLElement>(`[data-menu-item="${value}"]`);
  const railMenuTrigger = () => rail()?.querySelector<HTMLElement>('[aria-label="More actions"]');
  // The trigger's data-state belongs to its tooltip; the menu's own state is aria-expanded. The menu opens on a
  // later frame, so read it only once its items are in the document.
  const openRailMenu = async () => {
    const trigger = railMenuTrigger();
    expect(trigger).not.toBeNull();

    if (trigger?.getAttribute('aria-expanded') !== 'true') {
      await click(trigger);
    }

    await vi.waitFor(
      () => {
        expect(railMenuTrigger()?.getAttribute('aria-expanded')).toBe('true');
        expect(document.querySelector('[data-menu-item]')).not.toBeNull();
      },
      { timeout: 2000 }
    );
  };
  // Escape only reaches the menu once it holds focus, which lands a frame later; toggling the trigger closes it
  // regardless. A closed menu keeps its items mounted, so aria-expanded is what says it has closed.
  const closeRailMenu = async () => {
    if (railMenuTrigger()?.getAttribute('aria-expanded') === 'true') {
      await click(railMenuTrigger());
    }

    await vi.waitFor(() => expect(railMenuTrigger()?.getAttribute('aria-expanded')).toBe('false'), { timeout: 2000 });
  };
  const clickRailMenuItem = async (value: string) => {
    await openRailMenu();
    await click(menuItem(value));
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    browse.ensureWorkflowLibraryBrowseLoaded.mockClear();
    browse.setWorkflowLibraryBrowseFilter.mockClear();
    loader.open.mockClear();
    loader.resume.mockClear();
    loader.phase.current = 'idle';
    for (const command of Object.values(COMMANDS)) {
      command.mockClear();
    }
    project.setSnapshot(projectSnapshot(PROJECT_WORKFLOWS));
    browse.setSnapshot?.(LOADED_SNAPSHOT);
    accountLifecycle.invalidate();
    // The library opens on the project's own workflows, as it does from the workflow header.
    openWorkflowLibraryAtProjectWorkflow(PROJECT_ID, 'wf-alpha');
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('lists every project workflow as a card, badges the active one, and never fetches templates', async () => {
    await renderDialog();

    expect(document.querySelector('[data-library-tab="project"]')).not.toBeNull();
    expect(cards().map((element) => element.dataset.workflowCard)).toEqual(['wf-alpha', 'wf-bundled', 'wf-user']);
    expect(document.body.textContent).toContain('3 workflows in this project');

    const badges = [...document.querySelectorAll<HTMLElement>('[data-workflow-card] [data-active-workflow]')];
    expect(badges).toHaveLength(1);
    expect(badges[0]?.closest<HTMLElement>('[data-workflow-card]')?.dataset.workflowCard).toBe('wf-alpha');
    expect(badges[0]?.textContent).toBe('Active');

    // The project view is local; no library page is requested and no search is offered.
    expect(browse.ensureWorkflowLibraryBrowseLoaded).not.toHaveBeenCalled();
    expect(document.querySelector('input[type="search"]')).toBeNull();
  });

  it('starts a fresh workflow from the header', async () => {
    await renderDialog();

    await clickText('New workflow');

    expect(COMMANDS.createWorkflow).toHaveBeenCalledTimes(1);
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  it('moves to the bundled templates when asked to add a workflow', async () => {
    await renderDialog();

    await clickText('Add workflow');

    expect(workflowUiStore.getSnapshot().libraryTab).toBe('default');
    expect(browse.setWorkflowLibraryBrowseFilter).toHaveBeenCalledWith({ category: 'default', tag: null });
    expect(document.querySelector('[data-library-tab="default"]')).not.toBeNull();
    expect(browse.ensureWorkflowLibraryBrowseLoaded).toHaveBeenCalled();
  });

  it('opens a workflow from a double-click or the rail, then closes', async () => {
    await renderDialog();

    await act(() => card('wf-user')?.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));

    expect(COMMANDS.selectWorkflow).toHaveBeenCalledWith('wf-user');
    expect(onOpenChange).toHaveBeenCalledWith(false);

    await selectCard('wf-bundled');
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-bundled');

    await clickText('Open');

    expect(COMMANDS.selectWorkflow).toHaveBeenLastCalledWith('wf-bundled');
    expect(onOpenChange).toHaveBeenCalledTimes(2);
  });

  it("opens a project tile's actions from its menu button, including rename", async () => {
    await renderDialog();

    const button = document.querySelector<HTMLButtonElement>('[data-workflow-card-menu="wf-bundled"]');
    await act(async () => {
      button?.click();
      await settleFrame();
    });

    await vi.waitFor(() =>
      expect(document.querySelector('[data-workflow-context-menu] [data-menu-item="rename"]')).not.toBeNull()
    );
    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-bundled');
  });

  it('hands focus to the rail menu it opens, so the pointer can move onto it without closing it', async () => {
    await renderDialog();

    await click(railMenuTrigger());

    // Inside the dialog a menu that never took focus closes on the first pointer move onto it.
    await vi.waitFor(() => expect(document.activeElement?.closest('[role="menu"]')).not.toBeNull(), { timeout: 2000 });
  });

  it('shows the active workflow in the rail with Open disabled, since it is already open', async () => {
    await renderDialog();

    expect(rail()?.dataset.projectWorkflowDetail).toBe('wf-alpha');
    expect((buttonWithText('Active') as HTMLButtonElement | undefined)?.disabled).toBe(true);
    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe('Not linked to a library template');
  });

  it('renames through the dialog', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('rename');

    const input = document.querySelector<HTMLInputElement>('input[name="renameValue"]');
    expect(input?.value).toBe('Bundled copy');

    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await act(async () => {
      setValue?.call(input, 'Bundled, tuned');
      input?.dispatchEvent(new Event('input', { bubbles: true }));
      await Promise.resolve();
    });
    await click(input?.closest('form')?.querySelector<HTMLButtonElement>('button[type="submit"]'));

    expect(COMMANDS.renameWorkflow).toHaveBeenCalledWith('wf-bundled', 'Bundled, tuned');
  });

  it('duplicates beside the original under a copy name', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('duplicate');

    expect(COMMANDS.duplicateWorkflow).toHaveBeenCalledWith('wf-bundled', 'Bundled copy copy');
    // The copy becomes the selection so the rail shows what was just made.
    expect(workflowUiStore.getSnapshot().librarySelection).toEqual({ projectId: PROJECT_ID, workflowId: 'wf-copy' });
  });

  it('removes only after the confirmation is accepted', async () => {
    await renderDialog();
    await selectCard('wf-bundled');

    await clickRailMenuItem('remove');

    expect(alertDialog()?.textContent).toContain('Remove workflow');
    expect(alertDialog()?.textContent).toContain('Remove "Bundled copy" from this project?');
    expect(COMMANDS.removeWorkflow).not.toHaveBeenCalled();

    const confirm = [...(alertDialog()?.querySelectorAll('button') ?? [])].find(
      (candidate) => (candidate.textContent ?? '').trim() === 'Remove'
    );
    await click(confirm);

    expect(COMMANDS.removeWorkflow).toHaveBeenCalledWith('wf-bundled');
  });

  it('offers a template update only for a source the account can write', async () => {
    await renderDialog();
    await selectCard('wf-bundled');
    await openRailMenu();

    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe('From a bundled template (read-only)');
    expect(menuItem('save-to-library')).not.toBeNull();
    expect(menuItem('update-template')).toBeNull();

    await closeRailMenu();
    await selectCard('wf-user');
    await openRailMenu();

    expect(rail()?.querySelector('[data-workflow-source]')?.textContent).toBe(
      'Linked to library template at revision 4'
    );
    expect(menuItem('update-template')).not.toBeNull();

    await click(menuItem('update-template'));

    expect(workflowUiStore.getSnapshot().publicationIntent).toEqual({ kind: 'update-source', workflowId: 'wf-user' });
  });

  it('previews a project workflow from its own document', async () => {
    await renderDialog();
    await selectCard('wf-user');

    await clickText('Preview graph');
    await act(async () => {
      await vi.waitFor(() => expect(document.querySelector('[data-preview-dialog]')).not.toBeNull(), {
        timeout: 2000,
      });
    });

    const preview = document.querySelector('[data-preview-dialog]');
    expect(preview?.getAttribute('data-graph-id')).toBe('wf-user');
    expect(preview?.getAttribute('data-source-label')).toBe('User copy');
    expect(preview?.textContent).toContain('integer');
  });
});
