import type { InvocationTemplatesSnapshot } from '@features/workflow/core/types';
import type {
  GraphPreviewSourceState,
  WorkflowInvocationSourceId,
  WorkflowPreviewGraph,
} from '@features/workflow/ui/contracts';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { GraphPreviewDialog } from './GraphPreviewDialog';

// `downloadText`, the invocation templates snapshot, and the flow's `onInit`
// instance stub all need to exist before `vi.mock` factories below run
// (they're hoisted above the imports that would otherwise define them), so
// they're built through `vi.hoisted`.
const { downloads, fitViewMock, TEMPLATES_SNAPSHOT, templatesSnapshotRef } = vi.hoisted(() => {
  const fieldInput = (name: string, defaultValue: unknown) => ({
    default: defaultValue,
    description: '',
    exclusiveMaximum: null,
    exclusiveMinimum: null,
    fieldKind: 'input' as const,
    input: 'any' as const,
    maximum: null,
    minimum: null,
    multipleOf: null,
    name,
    options: null,
    required: false,
    title: name,
    type: { batch: false, cardinality: 'SINGLE' as const, name: 'IntegerField' },
    uiChoiceLabels: null,
    uiComponent: null,
    uiHidden: false,
    uiModelBase: null,
    uiModelFormat: null,
    uiModelType: null,
    uiOrder: null,
  });
  const invocationTemplate = (type: string, inputs: Record<string, ReturnType<typeof fieldInput>>) => ({
    category: 'test',
    classification: 'stable',
    description: '',
    inputs,
    nodePack: 'invokeai',
    outputs: {},
    outputType: `${type}_output`,
    tags: [],
    title: type,
    type,
    useCache: true,
    version: '1.0.0',
  });

  // Typed against the real snapshot shape (not inferred from this literal)
  // so `templatesSnapshotRef.current` can be reassigned to other statuses
  // (e.g. 'loading') from a test without a structural mismatch.
  const templatesSnapshot: InvocationTemplatesSnapshot = {
    error: null,
    status: 'loaded',
    templates: {
      denoise_latents: invocationTemplate('denoise_latents', {
        cfg_scale: fieldInput('cfg_scale', 7),
        steps: fieldInput('steps', 30),
      }),
      integer: invocationTemplate('integer', { value: fieldInput('value', 0) }),
      l2i: invocationTemplate('l2i', {}),
    },
  };

  return {
    downloads: { downloadBlob: vi.fn(), downloadText: vi.fn() },
    // `GraphPreviewFlow`'s mock (below) calls `onInit` with this so
    // `handleFlowInit`'s pending-reveal consumption has a real `fitView` spy
    // to assert against — asserting the dialog's own reveal logic runs to
    // completion, not just that it doesn't crash.
    fitViewMock: vi.fn(() => Promise.resolve(true)),
    TEMPLATES_SNAPSHOT: templatesSnapshot,
    // Mutable so a single test can point `useInvocationTemplatesSnapshot` at
    // a non-loaded status without a per-test `vi.mock` factory.
    templatesSnapshotRef: { current: templatesSnapshot },
  };
});

// xyflow stays out of this shell test — the flow pane's own rendering is
// covered elsewhere (`GraphPreviewFlow`'s own tests). It still calls `onInit`
// with a stub instance so `GraphPreviewDialog`'s pending-reveal handoff
// (`selectAndReveal` → `handleFlowInit`) has something real to run against.
vi.mock('./GraphPreviewFlow', () => ({
  GraphPreviewFlow: ({ onInit }: { onInit?: (instance: { fitView: typeof fitViewMock }) => void }) => {
    onInit?.({ fitView: fitViewMock });
    return <div data-flow-stub />;
  },
  documentToPreviewGraph: () => {
    throw new Error('not used');
  },
}));

vi.mock('@platform/browser/downloadBlob', () => downloads);

// The "Open as" menu (`GraphPreviewOpenAsMenu`) reads the invocation
// templates snapshot through the reactive hook, not the plain getter, so its
// disabled state can update live if the menu opens while templates are still
// loading — this stubs that hook (and the getter, for symmetry) with a
// pre-loaded snapshot covering the fixture graph's node types.
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => templatesSnapshotRef.current,
  useInvocationTemplatesSnapshot: () => templatesSnapshotRef.current,
}));

// "Save to workflow library" goes through `useSaveWorkflowToLibrary`
// (Task 8), which calls the backend through this barrel — stub the one
// function that path reaches, same mock shape as
// `useSaveWorkflowToLibrary.browser.test.tsx`.
const { createLibraryWorkflowMock } = vi.hoisted(() => ({
  createLibraryWorkflowMock: vi.fn(),
}));

vi.mock('@features/workflow/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  createLibraryWorkflow: createLibraryWorkflowMock,
}));

// The real client fetches en.json over HTTP (`platform/i18n/client.ts`), which
// this browser test never boots. Stub `t` with the subset of English strings
// this dialog renders, so assertions check real copy instead of raw dotted
// keys — the repo-wide convention for `react-i18next` in browser tests.
const TRANSLATIONS: Record<string, string> = {
  'common.close': 'Close',
  'common.json': 'JSON',
  'graphPreview.back': 'Back',
  'graphPreview.compiledFrom': 'Compiled from {{source}}.',
  'graphPreview.copied': 'Copied',
  'graphPreview.copyFailed': 'Failed to copy JSON',
  'graphPreview.copyJson': 'Copy JSON',
  'graphPreview.destination': 'Destination',
  'graphPreview.downloadJson': 'Download JSON',
  'graphPreview.downloadJsonHint': 'For bug reports and sharing',
  'graphPreview.edges': 'Edges',
  'graphPreview.edgesIn': 'in · {{count}} inputs from {{sources}}',
  'graphPreview.edgesInNone': 'in · none',
  'graphPreview.edgesOut': 'out · {{field}} → {{target}}',
  'graphPreview.editInEditor': 'Edit in workflow editor',
  'graphPreview.editInEditorFailed': 'No editable nodes in this graph.',
  'graphPreview.editInEditorHint': 'Replaces the current workflow',
  'graphPreview.forkIntoProject': 'Fork into new project',
  'graphPreview.forkIntoProjectFailed': 'No nodes to fork into a project.',
  'graphPreview.forkIntoProjectHint': 'Copies this graph into a fresh project',
  'graphPreview.graph': 'Graph',
  'graphPreview.graphJsonLabel': '{{title}} graph JSON',
  'graphPreview.inputCount': '{{count}} inputs',
  'graphPreview.invalidTitle': "This graph can't compile yet.",
  'graphPreview.invokeRoute': 'Invoke {{route}}',
  'graphPreview.list': 'List',
  'graphPreview.liveHint': 'Updates as you change settings.',
  'graphPreview.noCompiledGraph': 'No compiled graph is available for "{{graphId}}" yet.',
  'graphPreview.nodeSummary.denoise': '{{steps}} steps · cfg {{cfg}}',
  'graphPreview.nodeSummary.noise': '{{width}} × {{height}}',
  'graphPreview.nodes': 'Nodes',
  'graphPreview.openAs': 'Open as',
  'graphPreview.openedFromPreview': 'Opened from graph preview',
  'graphPreview.resolvedInputs': 'Resolved inputs',
  'graphPreview.savedToLibrary': 'Saved to workflow library',
  'graphPreview.saveToLibrary': 'Save to workflow library',
  'graphPreview.saveToLibraryFailed': 'No saveable nodes in this graph.',
  'graphPreview.saveToLibraryHint': 'Reusable, leaves this project alone',
  'graphPreview.selectNode': 'Select a node for details.',
  'graphPreview.setBy': 'Set by',
  'graphPreview.showNode': 'show node',
  'graphPreview.thisGraph': 'This graph',
  'graphPreview.title': 'Graph preview',
  'workflowLibrary.saveFailed': 'Failed to save workflow',
};

const interpolate = (template: string, options?: Record<string, unknown>): string =>
  options ? template.replace(/\{\{(\w+)\}\}/g, (_match, key: string) => String(options[key] ?? '')) : template;

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => interpolate(TRANSLATIONS[key] ?? key, options),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const FIXTURE_GRAPH: WorkflowPreviewGraph = {
  id: 'preview-graph',
  nodes: [
    { id: 'seed', type: 'integer', inputs: { value: 42 } },
    { id: 'denoise_latents', type: 'denoise_latents', inputs: { cfg_scale: 4, steps: 28 } },
    { id: 'l2i', type: 'l2i', inputs: {} },
  ],
  edges: [
    { id: 'e1', sourceField: 'value', sourceNodeId: 'seed', targetField: 'seed', targetNodeId: 'denoise_latents' },
    {
      id: 'e2',
      sourceField: 'latents',
      sourceNodeId: 'denoise_latents',
      targetField: 'latents',
      targetNodeId: 'l2i',
    },
  ],
  version: 1,
};

const FIXTURE_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  getProvenance: (nodeId, fieldName) => {
    if (nodeId === 'denoise_latents' && fieldName === 'steps') {
      return { label: 'Generate → Steps' };
    }

    if (nodeId === 'denoise_latents' && fieldName === 'cfg_scale') {
      return { label: 'Generate → CFG scale' };
    }

    return null;
  },
  graph: FIXTURE_GRAPH,
  invalidReasons: [],
  isLive: true,
  notices: [
    { id: 'seed-random', message: 'Seed is randomized. This graph runs differently each time.', nodeId: 'seed' },
  ],
  resolvedInputOverrides: { seed: { value: 'regenerated each run' } },
  summaryRows: [
    { id: 'steps', label: 'Steps', value: '28' },
    { id: 'model', label: 'Model', value: 'SDXL' },
  ],
};

/** A library entry: previewed before it has been opened into a project, so nothing has routed it anywhere. */
const NO_DESTINATION_SOURCE: GraphPreviewSourceState = {
  destinationLabel: null,
  graph: FIXTURE_GRAPH,
  invalidReasons: [],
  isLive: false,
  notices: [],
  summaryRows: [],
};

const INVALID_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  graph: null,
  invalidReasons: ['Height must be a multiple of 8.'],
  isLive: true,
  notices: [],
  summaryRows: [],
};

// A single node of a type absent from `TEMPLATES_SNAPSHOT.templates` —
// `previewGraphToDocument` skips it, so the converted document has zero
// nodes and both "Open as" actions that convert to a document should bail.
const UNKNOWN_NODE_GRAPH: WorkflowPreviewGraph = {
  id: 'unknown-node-graph',
  nodes: [{ id: 'mystery', type: 'unknown_type', inputs: {} }],
  edges: [],
  version: 1,
};

const UNKNOWN_NODE_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  graph: UNKNOWN_NODE_GRAPH,
  invalidReasons: [],
  isLive: false,
  notices: [],
  summaryRows: [],
};

// Long enough to exercise the side panel's ~40-char truncation policy for
// resolved-input strings.
const LONG_STRING_VALUE =
  'a value long enough to need truncation in the resolved inputs list, well past forty characters';

const LONG_VALUE_GRAPH: WorkflowPreviewGraph = {
  id: 'long-value-graph',
  nodes: [{ id: 'note', type: 'integer', inputs: { value: LONG_STRING_VALUE } }],
  edges: [],
  version: 1,
};

const LONG_VALUE_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  graph: LONG_VALUE_GRAPH,
  invalidReasons: [],
  isLive: false,
  notices: [],
  summaryRows: [],
};

const preferencesSnapshot = {
  reduceMotion: false,
  themeId: 'classic' as const,
  workflowEdgeStyle: 'curved' as const,
  workflowShowMinimap: true,
  workflowSnapToGrid: false,
  workflowValidateConnections: true,
};

// A stable object per adapter instance — `useWorkflowProjectSelector`
// (`useSaveWorkflowToLibrary`, called by `GraphPreviewOpenAsMenu`) runs this
// through `useSyncExternalStoreWithSelector`'s shallow-equality check, which
// would otherwise see a "changed" snapshot on every render (new object
// identity, same content) and force an infinite re-render loop — the same
// reason `preferencesSnapshot` above is hoisted rather than built inline.
const createProjectSnapshot = () => ({
  galleryValues: {},
  graphHistory: [],
  id: 'project-1',
  isWorkflowRunning: false,
  projectGraph: { edges: [], nodes: [], version: 1 as const },
  workflowValues: {},
});

const createWorkflowUiAdapter = (): WorkflowUiAdapter => {
  const projectSnapshot = createProjectSnapshot();

  return {
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
    commands: {
      bindLibraryWorkflow: vi.fn(),
      editGraph: vi.fn(),
      redo: vi.fn(),
      replace: vi.fn(),
      restoreSnapshot: vi.fn(),
      saveSnapshot: vi.fn(),
      undo: vi.fn(),
    },
    getProjectGraph: () => ({ edges: [], nodes: [], version: 1 as const }),
    nodeExecution: { get: () => null, subscribe: () => () => {} },
    notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
    performance: {
      mark: vi.fn(),
      measure: vi.fn(),
      time: (_name: string, _source: unknown, callback: () => unknown) => callback(),
    },
    preferences: { getSnapshot: () => preferencesSnapshot, subscribe: () => () => {} },
    project: {
      getSnapshot: () => projectSnapshot,
      subscribe: () => () => {},
    },
    registerModalHotkeyLayer: vi.fn(() => vi.fn()),
    widgets: { open: vi.fn(), patchValues: vi.fn() },
  } as unknown as WorkflowUiAdapter;
};

const createGraphPreviewPort = (): WorkflowGraphPreviewPort => ({
  focusSource: vi.fn(),
  getRoute: () => ({ canInvoke: true, label: 'Generate → Gallery' }),
  invoke: vi.fn(() => Promise.resolve(true)),
  openDocumentInNewProject: vi.fn(),
  openWorkflowEditor: vi.fn(),
});

describe('GraphPreviewDialog', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;
  let onExitComplete: () => void;
  let graphPreviewPort: WorkflowGraphPreviewPort;
  let workflowUiAdapter: WorkflowUiAdapter;

  const dialogTree = (
    source: GraphPreviewSourceState,
    isOpen: boolean,
    sourceId: WorkflowInvocationSourceId,
    hideInvoke: boolean
  ) => (
    <StrictMode>
      <ChakraProvider value={system}>
        <WorkflowUiProvider adapter={workflowUiAdapter}>
          <WorkflowGraphPreviewProvider adapter={graphPreviewPort}>
            <GraphPreviewDialog
              graphId="preview-graph-id"
              hideInvoke={hideInvoke}
              isOpen={isOpen}
              source={source}
              sourceId={sourceId}
              sourceLabel="Generate"
              onExitComplete={onExitComplete}
              onOpenChange={onOpenChange}
            />
          </WorkflowGraphPreviewProvider>
        </WorkflowUiProvider>
      </ChakraProvider>
    </StrictMode>
  );

  const renderDialog = async (
    source: GraphPreviewSourceState,
    isOpen = true,
    sourceId: WorkflowInvocationSourceId = 'generate',
    hideInvoke = false
  ) => {
    await act(() => {
      root.render(dialogTree(source, isOpen, sourceId, hideInvoke));
    });
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    onExitComplete = vi.fn();
    graphPreviewPort = createGraphPreviewPort();
    workflowUiAdapter = createWorkflowUiAdapter();
    downloads.downloadBlob.mockReset();
    downloads.downloadText.mockReset();
    createLibraryWorkflowMock.mockReset();
    fitViewMock.mockClear();
    templatesSnapshotRef.current = TEMPLATES_SNAPSHOT;
  });

  afterEach(async () => {
    await act(() => root.unmount());
    // The "Open as" menu's content portals to `document.body`, outside the
    // React root this suite unmounts above — sweep it so a leftover node
    // from one test can't answer a `[role="menuitem"]` query in the next.
    document.querySelectorAll('[data-scope="menu"]').forEach((element) => element.remove());
    host.remove();
  });

  const switchToMode = async (mode: 'graph' | 'list' | 'json') => {
    const input = document.querySelector<HTMLInputElement>(`input[value="${mode}"]`);
    expect(input).not.toBeNull();
    const label = input?.closest('label');
    expect(label).not.toBeNull();

    await act(() => {
      label?.click();
    });
  };

  const clickButtonWithText = async (text: string) => {
    const button = [...document.querySelectorAll('button')].find((candidate) =>
      (candidate.textContent ?? '').includes(text)
    );
    expect(button).not.toBeUndefined();

    await act(() => {
      button?.click();
    });
  };

  // Opens the "Open as" menu. Chakra's `Menu.Content` is lazy-mounted and
  // its open transition runs on a timer, so the click alone isn't enough —
  // give the portal a beat to attach before querying for items.
  const openAsMenu = async () => {
    await clickButtonWithText('Open as');
    await act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 40);
      });
    });
  };

  const findMenuItemWithText = (text: string) =>
    [...document.querySelectorAll('[role="menuitem"]')].find((candidate) =>
      (candidate.textContent ?? '').includes(text)
    );

  const clickMenuItemWithText = async (text: string) => {
    const item = findMenuItemWithText(text);
    expect(item).not.toBeUndefined();

    await act(() => {
      (item as HTMLElement | undefined)?.click();
    });
  };

  // "Save to workflow library" fires off an async handler (`void
  // saveToLibrary()`) that the click itself doesn't wait for — a macrotask
  // tick drains the `await saveDocumentAsNew(document)` chain (serialize →
  // `createLibraryWorkflow` → notify) before assertions run.
  const flushAsync = () =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });

  it('renders summary rows, node count, and the live subtitle', async () => {
    await renderDialog(FIXTURE_SOURCE);

    const text = document.body.textContent ?? '';

    expect(text).toContain('This graph');
    expect(text).toContain('Gallery');
    expect(text).toContain('3');
    expect(text).toContain('Updates as you change settings.');
  });

  it('omits the destination row when the source was never routed anywhere', async () => {
    await renderDialog(NO_DESTINATION_SOURCE);

    const panel = document.querySelector('[role="region"][aria-label="This graph"]');
    // A "Destination —" row states nothing; the node count still shows.
    expect(panel?.textContent ?? '').not.toContain('Destination');
    expect(panel?.textContent ?? '').toContain('Nodes');
  });

  it('reports when its close transition has finished, so hosts can drop the mount', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect(onExitComplete).not.toHaveBeenCalled();

    await renderDialog(FIXTURE_SOURCE, false);

    // The exit animation runs in real time; polling inside `act` keeps its
    // final commit inside an open act scope.
    await act(async () => {
      await vi.waitFor(() => {
        expect(onExitComplete).toHaveBeenCalled();
      });
    });
  });

  it('stays landscape on a tall viewport instead of growing into a full-height column', async () => {
    const original = { height: window.innerHeight, width: window.innerWidth };

    try {
      // Resized deliberately: at the default test viewport (~866px tall) an
      // 80vh dialog and a 46rem-capped one measure within pixels of each other,
      // so nothing asserted at that size can tell the two sizings apart.
      await page.viewport(1280, 1400);
      await renderDialog(FIXTURE_SOURCE);

      const content = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]');
      expect(content).not.toBeNull();

      const box = content?.getBoundingClientRect();

      // A viewport-proportional height would be ~1100px here; the cap is 46rem.
      expect(box?.height ?? 0).toBeLessThanOrEqual(46 * 16 + 1);
      // Which leaves the dialog the shape a graph reads in.
      expect(box?.width ?? 0).toBeGreaterThan(box?.height ?? 0);
    } finally {
      await page.viewport(original.width, original.height);
    }
  });

  it('shows the seed notice inline in the summary panel', async () => {
    await renderDialog(FIXTURE_SOURCE);

    const panel = document.querySelector('[role="region"][aria-label="This graph"]');
    expect(panel?.textContent ?? '').toContain('Seed is randomized');
  });

  it('switches to JSON mode', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect(document.querySelector('[data-flow-stub]')).not.toBeNull();

    const jsonInput = document.querySelector<HTMLInputElement>('input[value="json"]');
    expect(jsonInput).not.toBeNull();
    const jsonLabel = jsonInput?.closest('label');
    expect(jsonLabel).not.toBeNull();

    await act(() => {
      jsonLabel?.click();
    });

    expect(document.querySelector('[data-flow-stub]')).toBeNull();
    expect(document.body.textContent ?? '').toContain('"denoise_latents"');
  });

  it('shows the first invalid reason and no flow pane when compile is blocked', async () => {
    await renderDialog(INVALID_SOURCE);

    expect(document.body.textContent ?? '').toContain('Height must be a multiple of 8.');
    expect(document.querySelector('[data-flow-stub]')).toBeNull();
  });

  it('renders the side panel only in graph mode', async () => {
    // The fixture's own notice text ("...This graph runs differently each
    // time.") contains the panel's heading as a substring, so this checks
    // for the panel's `Scrollable` region (`aria-label="This graph"`)
    // instead of a raw text match.
    const findSidePanel = () => document.querySelector('[role="region"][aria-label="This graph"]');

    await renderDialog(FIXTURE_SOURCE);

    // Graph mode (the default): the panel is present.
    expect(findSidePanel()).not.toBeNull();

    await switchToMode('list');
    expect(findSidePanel()).toBeNull();

    await switchToMode('json');
    expect(findSidePanel()).toBeNull();

    await switchToMode('graph');
    expect(findSidePanel()).not.toBeNull();
  });

  it('disables Copy JSON when there is no compiled graph to copy', async () => {
    await renderDialog(INVALID_SOURCE);

    const copyButton = [...document.querySelectorAll('button')].find((candidate) =>
      (candidate.textContent ?? '').includes('Copy JSON')
    );

    expect(copyButton).toBeInstanceOf(HTMLButtonElement);
    expect((copyButton as HTMLButtonElement).disabled).toBe(true);
  });

  it('selecting a node from the list opens the inspector with resolved inputs and edges', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await switchToMode('list');
    expect(document.querySelector('[data-flow-stub]')).toBeNull();

    await clickButtonWithText('denoise_latents');

    // List selection reveals the node in graph mode, not list mode.
    expect(document.querySelector<HTMLInputElement>('input[value="graph"]')?.checked).toBe(true);
    expect(document.querySelector('[data-flow-stub]')).not.toBeNull();

    const text = document.body.textContent ?? '';
    expect(text).toContain('denoise_latents');
    expect(text).toContain('Resolved inputs');
    expect(text).toContain('28');
    expect(text).toContain('Set by');
    expect(text).toContain('Generate → Steps');

    // The fixture's `denoise_latents` node has one incoming edge (from `seed`)
    // and one outgoing edge (to `l2i`) — exercise both `getEdgesInLine` and
    // the per-edge `edgesOut` lines, not just that the "Edges" heading renders.
    expect(text).toContain('Edges');
    expect(text).toContain('in · 1 inputs from seed');
    expect(text).toContain('out · latents → l2i');

    // The flow wasn't mounted when the row was clicked (mode was still
    // 'list'), so the reveal had to go through the pending-reveal path:
    // `selectAndReveal` stashes the id, and the flow's remount (its `onInit`
    // firing again) is what actually calls `fitView`. This is the case the
    // stale-ref bug broke — before the fix, `flowInstanceRef` still pointed
    // at the unmounted flow's instance and this fit never happened.
    expect(fitViewMock).toHaveBeenCalledWith(expect.objectContaining({ nodes: [{ id: 'denoise_latents' }] }));
  });

  it('show node selects the seed node and inspector shows the randomized override', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await clickButtonWithText('show node');

    const text = document.body.textContent ?? '';
    expect(text).toContain('integer');
    expect(text).toContain('seed');
    expect(text).toContain('regenerated each run');

    // Mode was already 'graph' with the flow mounted, so this reveal takes
    // the immediate branch — `fitView` runs straight off `flowInstanceRef`
    // instead of waiting on a remount.
    expect(fitViewMock).toHaveBeenCalledWith(expect.objectContaining({ nodes: [{ id: 'seed' }] }));
  });

  it('clicking a provenance link focuses the source and closes the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await switchToMode('list');
    await clickButtonWithText('denoise_latents');
    await clickButtonWithText('Generate → Steps');

    expect(graphPreviewPort.focusSource).toHaveBeenCalledWith('generate');
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('clears the selected node when the dialog closes and reopens', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await switchToMode('list');
    await clickButtonWithText('denoise_latents');
    expect(document.body.textContent ?? '').toContain('Resolved inputs');

    // The footer Close button is the path `closeAndReset` resets `selectedNodeId`
    // through. Re-render with `isOpen` false then true to simulate the parent
    // obeying the `onOpenChange(false)` this just triggered and reopening —
    // the same controlled-`open` round trip a real host does.
    await clickButtonWithText('Close');
    expect(onOpenChange).toHaveBeenCalledWith(false);

    await renderDialog(FIXTURE_SOURCE, false);
    await renderDialog(FIXTURE_SOURCE, true);

    const text = document.body.textContent ?? '';
    expect(text).toContain('Select a node for details.');
    expect(text).not.toContain('Resolved inputs');
  });

  it('Open as → Edit in workflow editor replaces the document, opens the editor, closes the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Edit in workflow editor');

    expect(workflowUiAdapter.commands.replace).toHaveBeenCalledTimes(1);
    const [document_, label] = vi.mocked(workflowUiAdapter.commands.replace).mock.calls[0] ?? [];
    expect(document_?.nodes).toHaveLength(3);
    expect(label).toBe('Opened from graph preview');
    expect(graphPreviewPort.openWorkflowEditor).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('Open as → Fork into new project hands the named document to the port and closes the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Fork into new project');

    expect(graphPreviewPort.openDocumentInNewProject).toHaveBeenCalledTimes(1);
    const [document_, label] = vi.mocked(graphPreviewPort.openDocumentInNewProject).mock.calls[0] ?? [];
    expect(document_?.nodes).toHaveLength(3);
    // The fixture graph has no `label`, so this exercises the
    // `graph.label ?? sourceLabel` fallback (`sourceLabel` is "Generate").
    expect(document_?.name).toBe('Generate');
    expect(label).toBe('Opened from graph preview');
    // Forking must not touch the current project's workflow.
    expect(workflowUiAdapter.commands.replace).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('Open as → Download JSON downloads the backend graph', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Download JSON');

    expect(downloads.downloadText).toHaveBeenCalledTimes(1);
    const [content, fileName, type] = vi.mocked(downloads.downloadText).mock.calls[0] ?? [];
    // The fixture graph has no `backendGraph`, so this exercises the
    // `graph.backendGraph ?? graph` fallback, not just the happy path.
    expect(content).toContain('"denoise_latents"');
    expect(fileName).toBe('graph.json');
    expect(type).toBe('application/json');
  });

  it('hides Edit in workflow editor for the workflow source', async () => {
    await renderDialog(FIXTURE_SOURCE, true, 'workflow');

    await openAsMenu();

    expect(findMenuItemWithText('Edit in workflow editor')).toBeUndefined();
    expect(findMenuItemWithText('Save to workflow library')).not.toBeUndefined();
    expect(findMenuItemWithText('Download JSON')).not.toBeUndefined();
  });

  it('renders the footer Invoke button by default', async () => {
    await renderDialog(FIXTURE_SOURCE);

    const buttonText = [...document.querySelectorAll('button')].map((button) => button.textContent ?? '');
    expect(buttonText.some((text) => text.includes('Invoke Generate → Gallery'))).toBe(true);
  });

  it('hideInvoke hides only the footer Invoke button — Copy JSON and Open as stay', async () => {
    await renderDialog(FIXTURE_SOURCE, true, 'generate', true);

    const buttonText = [...document.querySelectorAll('button')].map((button) => button.textContent ?? '');
    expect(buttonText.some((text) => text.includes('Invoke'))).toBe(false);
    expect(buttonText.some((text) => text.includes('Copy JSON'))).toBe(true);
    expect(buttonText.some((text) => text.includes('Open as'))).toBe(true);
  });

  it('Open as → Save to workflow library names the document from the source label and notifies success', async () => {
    createLibraryWorkflowMock.mockResolvedValue('library-workflow-1');

    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowMock).toHaveBeenCalledTimes(1);
    // The fixture graph has no `label`, so this exercises the
    // `graph.label ?? sourceLabel` fallback (`sourceLabel` is "Generate").
    const [serialized] = createLibraryWorkflowMock.mock.calls[0] ?? [];
    expect(serialized).toMatchObject({ name: 'Generate' });
    expect(workflowUiAdapter.notifications.success).toHaveBeenCalledWith('Saved to workflow library');
  });

  it('Open as → Save to workflow library does not notify success when the save fails', async () => {
    createLibraryWorkflowMock.mockRejectedValue(new Error('network down'));

    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowMock).toHaveBeenCalledTimes(1);
    expect(workflowUiAdapter.notifications.success).not.toHaveBeenCalled();
    // `useSaveWorkflowToLibrary`'s own catch path (Task 8) reports the failure.
    expect(workflowUiAdapter.notifications.error).toHaveBeenCalledWith('Failed to save workflow', expect.any(String));
  });

  it('Open as → Save to workflow library bails with an error notification when the graph has no saveable nodes', async () => {
    await renderDialog(UNKNOWN_NODE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    // The fixture's only node is a type with no matching template, so
    // `previewGraphToDocument` skips it and the converted document is empty —
    // this must bail before ever reaching the backend save call.
    expect(createLibraryWorkflowMock).not.toHaveBeenCalled();
    expect(workflowUiAdapter.notifications.error).toHaveBeenCalledWith('No saveable nodes in this graph.');
    expect(workflowUiAdapter.notifications.success).not.toHaveBeenCalled();
  });

  it('disables "Save to workflow library" while invocation templates are still loading', async () => {
    templatesSnapshotRef.current = { error: null, status: 'loading', templates: {} };

    await renderDialog(FIXTURE_SOURCE);
    await openAsMenu();

    const item = findMenuItemWithText('Save to workflow library');
    expect(item).not.toBeUndefined();
    expect(item?.getAttribute('aria-disabled')).toBe('true');
  });

  it('truncates a long resolved-input string and exposes the full value via title', async () => {
    await renderDialog(LONG_VALUE_SOURCE);

    await switchToMode('list');
    await clickButtonWithText('note');

    const truncated = `${LONG_STRING_VALUE.slice(0, 40)}…`;
    const valueElement = [...document.querySelectorAll('dd')].find((element) => element.textContent === truncated);

    expect(valueElement).not.toBeUndefined();
    expect(valueElement?.getAttribute('title')).toBe(LONG_STRING_VALUE);
    // The untruncated value never appears verbatim in the rendered text.
    expect(document.body.textContent ?? '').not.toContain(LONG_STRING_VALUE);
  });
});
