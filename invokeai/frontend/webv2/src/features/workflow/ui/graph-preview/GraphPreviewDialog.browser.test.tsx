import type { InvocationTemplatesSnapshot } from '@features/workflow/core/types';
import type {
  GraphPreviewSourceState,
  WorkflowInvocationSourceId,
  WorkflowPreviewGraph,
} from '@features/workflow/ui/contracts';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { GraphPreviewDialog } from './GraphPreviewDialog';

// Create mock dependencies with vi.hoisted because factories run before ordinary imports.
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
    // Provide an initialized flow spy to verify pending reveals actually call fitView.
    fitViewMock: vi.fn(() => Promise.resolve(true)),
    TEMPLATES_SNAPSHOT: templatesSnapshot,
    templatesSnapshotRef: { current: templatesSnapshot },
  };
});

// Stub flow rendering but retain onInit to exercise the dialog's pending-reveal handoff.
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

// Stub reactive templates so Open as disabled state follows the same loading contract as production.
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => templatesSnapshotRef.current,
  useInvocationTemplatesSnapshot: () => templatesSnapshotRef.current,
}));

const { createLibraryWorkflowRecordMock } = vi.hoisted(() => ({
  createLibraryWorkflowRecordMock: vi.fn(),
}));

// The publication controller reaches the transport through the API module itself, not the queries barrel.
vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  createLibraryWorkflowRecord: createLibraryWorkflowRecordMock,
}));

// Supply rendered English strings without starting the HTTP-backed i18n client.
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
  'workflowLibrary.saved': 'Workflow saved',
  'workflowLibrary.savedCreatedBody': 'Saved "{{name}}" to the library.',
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

// An unknown-only graph converts to an empty document; document-opening actions must stop.
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
  workflowEdgesBehindNodes: false,
  workflowShowMinimap: true,
  workflowSnapToGrid: false,
  workflowValidateConnections: true,
};

// Keep adapter snapshots stable so external-store subscribers cannot loop on fresh equivalent objects.
const createProjectSnapshot = () => {
  const projectGraph = createProjectGraph('workflow-1');

  return {
    activeWorkflow: { document: projectGraph },
    activeWorkflowId: projectGraph.id,
    galleryValues: {},
    id: 'project-1',
    isWorkflowRunning: false,
    projectGraph,
    workflowValues: {},
    workflows: [{ document: projectGraph }],
  };
};

const createWorkflowUiAdapter = (): WorkflowUiAdapter => {
  const projectSnapshot = createProjectSnapshot();

  return {
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
    commands: {
      addWorkflow: vi.fn(),
      createWorkflow: vi.fn(),
      duplicateWorkflow: vi.fn(),
      editGraph: vi.fn(),
      redo: vi.fn(),
      removeWorkflow: vi.fn(),
      renameWorkflow: vi.fn(),
      selectWorkflow: vi.fn(),
      setWorkflowSource: vi.fn(),
      undo: vi.fn(),
    },
    getProjectGraph: () => projectSnapshot.projectGraph,
    nodeExecution: {
      get: () => null,
      getOrigin: () => null,
      subscribe: () => () => {},
      subscribeOrigin: () => () => {},
    },
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
    createLibraryWorkflowRecordMock.mockReset();
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
    const tab = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((candidate) =>
      candidate.id.endsWith(`-tab-${mode}`)
    );
    expect(tab).not.toBeUndefined();

    await act(() => {
      tab?.click();
    });
    // List mode loads on demand; wait for its rows before acting on them.
    if (mode === 'list') {
      await expect.element(page.getByRole('list').getByRole('button').first()).toBeVisible();
    }
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

  // Wait for lazy menu portal mounting before querying items.
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

  // Await the detached save handler's async chain before asserting serialization, creation, and notification.
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

    await switchToMode('json');

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
    const graphTab = [...document.querySelectorAll<HTMLElement>('[role="tab"]')].find((candidate) =>
      candidate.id.endsWith('-tab-graph')
    );
    expect(graphTab?.getAttribute('aria-selected')).toBe('true');
    expect(document.querySelector('[data-flow-stub]')).not.toBeNull();

    const text = document.body.textContent ?? '';
    expect(text).toContain('denoise_latents');
    expect(text).toContain('Resolved inputs');
    expect(text).toContain('28');
    expect(text).toContain('Set by');
    expect(text).toContain('Generate → Steps');

    // Exercise both incoming and outgoing edge details, not merely their heading.
    expect(text).toContain('Edges');
    expect(text).toContain('in · 1 inputs from seed');
    expect(text).toContain('out · latents → l2i');

    // List-mode reveal must wait for the remounted flow's onInit rather than use the destroyed previous instance.
    expect(fitViewMock).toHaveBeenCalledWith(expect.objectContaining({ nodes: [{ id: 'denoise_latents' }] }));
  });

  it('lists nodes as a named list and marks the selected node when returning to list mode', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await switchToMode('list');
    const list = () => document.querySelector<HTMLElement>('[role="list"][aria-label="Nodes"]');

    expect(list()).not.toBeNull();
    expect(list()!.querySelectorAll('[role="listitem"]').length).toBeGreaterThan(1);
    expect(list()!.querySelector('[aria-current="true"]')).toBeNull();

    await clickButtonWithText('denoise_latents');
    await switchToMode('list');

    const current = list()!.querySelector<HTMLElement>('[data-list-primary][aria-current="true"]');

    expect(current?.textContent).toContain('denoise_latents');
    // The selected node is the list's single tab stop, so keyboard users land on it.
    expect(current?.tabIndex).toBe(0);
    expect(list()!.querySelectorAll('[data-list-primary][tabindex="0"]')).toHaveLength(1);
  });

  it('show node selects the seed node and inspector shows the randomized override', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await clickButtonWithText('show node');

    const text = document.body.textContent ?? '';
    expect(text).toContain('integer');
    expect(text).toContain('seed');
    expect(text).toContain('regenerated each run');

    // Graph-mode reveal uses the already-mounted instance immediately.
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

    // Model the controlled close/reopen round trip to verify selection resets.
    await clickButtonWithText('Close');
    expect(onOpenChange).toHaveBeenCalledWith(false);

    await renderDialog(FIXTURE_SOURCE, false);
    await renderDialog(FIXTURE_SOURCE, true);

    const text = document.body.textContent ?? '';
    expect(text).toContain('Select a node for details.');
    expect(text).not.toContain('Resolved inputs');
  });

  it('Open as → Edit in workflow editor adds the document as a project workflow, opens the editor, closes the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Edit in workflow editor');

    expect(workflowUiAdapter.commands.addWorkflow).toHaveBeenCalledTimes(1);
    const [document_, options] = vi.mocked(workflowUiAdapter.commands.addWorkflow).mock.calls[0] ?? [];
    expect(document_?.nodes).toHaveLength(3);
    // A blank starter workflow may be taken over; an edited one gets the preview beside it.
    expect(options).toEqual({ label: 'Opened from graph preview', reusePlaceholder: true });
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
    expect(document_?.name).toBe('Generate');
    expect(label).toBe('Opened from graph preview');
    // Forking must not touch the current project's workflows.
    expect(workflowUiAdapter.commands.addWorkflow).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('Open as → Download JSON downloads the backend graph', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Download JSON');

    expect(downloads.downloadText).toHaveBeenCalledTimes(1);
    const [content, fileName, type] = vi.mocked(downloads.downloadText).mock.calls[0] ?? [];
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

  it('keeps the footer Invoke button inert while the route cannot be invoked', async () => {
    graphPreviewPort = {
      ...createGraphPreviewPort(),
      getRoute: () => ({ canInvoke: false, label: 'Generate → Gallery', validationMessage: 'Preparing generation…' }),
    };
    await renderDialog(FIXTURE_SOURCE);

    const button = [...document.querySelectorAll('button')].find((candidate) =>
      (candidate.textContent ?? '').includes('Invoke Generate → Gallery')
    );
    expect(button?.getAttribute('aria-disabled')).toBe('true');
    expect(button?.getAttribute('title')).toBe('Preparing generation…');

    await clickButtonWithText('Invoke Generate → Gallery');

    expect(graphPreviewPort.invoke).not.toHaveBeenCalled();
    expect(onOpenChange).not.toHaveBeenCalled();
  });

  it('hideInvoke hides only the footer Invoke button — Copy JSON and Open as stay', async () => {
    await renderDialog(FIXTURE_SOURCE, true, 'generate', true);

    const buttonText = [...document.querySelectorAll('button')].map((button) => button.textContent ?? '');
    expect(buttonText.some((text) => text.includes('Invoke'))).toBe(false);
    expect(buttonText.some((text) => text.includes('Copy JSON'))).toBe(true);
    expect(buttonText.some((text) => text.includes('Open as'))).toBe(true);
  });

  it('Open as → Save to workflow library names the document from the source label and notifies success', async () => {
    createLibraryWorkflowRecordMock.mockImplementation((workflow: Record<string, unknown>) =>
      Promise.resolve({ name: workflow.name, revision: 1, workflow, workflow_id: 'library-workflow-1' })
    );

    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(1);
    const [serialized, options] = createLibraryWorkflowRecordMock.mock.calls[0] ?? [];
    expect(serialized).toMatchObject({ name: 'Generate' });
    // A client-reserved id makes a resend after a lost answer idempotent.
    expect(options).toMatchObject({ reservedId: expect.any(String) });
    expect(workflowUiAdapter.notifications.success).toHaveBeenCalledWith(
      'Workflow saved',
      'Saved "Generate" to the library.'
    );
    // A preview is not a project workflow, so nothing in the project is pointed at the new template.
    expect(workflowUiAdapter.commands.setWorkflowSource).not.toHaveBeenCalled();
  });

  it('Open as → Save to workflow library retries a lost answer once under the same reserved id, never claiming success', async () => {
    createLibraryWorkflowRecordMock.mockRejectedValue(new Error('network down'));

    await renderDialog(FIXTURE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(2);
    const [, first] = createLibraryWorkflowRecordMock.mock.calls[0] ?? [];
    const [, second] = createLibraryWorkflowRecordMock.mock.calls[1] ?? [];
    expect((second as { reservedId: string }).reservedId).toBe((first as { reservedId: string }).reservedId);
    expect(workflowUiAdapter.notifications.success).not.toHaveBeenCalled();
  });

  it('Open as → Save to workflow library bails with an error notification when the graph has no saveable nodes', async () => {
    await renderDialog(UNKNOWN_NODE_SOURCE);

    await openAsMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    // Reject conversion with no recognized nodes before backend save.
    expect(createLibraryWorkflowRecordMock).not.toHaveBeenCalled();
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
