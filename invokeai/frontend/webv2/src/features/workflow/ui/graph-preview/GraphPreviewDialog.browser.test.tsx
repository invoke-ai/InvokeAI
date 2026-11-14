import type {
  InvocationTemplatesSnapshot,
  ProjectGraphState,
  WorkflowInvocationNode,
} from '@features/workflow/core/types';
import type {
  GraphPreviewInvoke,
  GraphPreviewSourceState,
  WorkflowInvocationSourceId,
  WorkflowPreviewGraph,
} from '@features/workflow/ui/contracts';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import {
  getUnresolvedWorkflowPublication,
  getWorkflowPublicationController,
} from '@features/workflow/ui/library/useWorkflowPublication';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { GraphPreviewDialog } from './GraphPreviewDialog';

// Create mock dependencies with vi.hoisted because factories run before ordinary imports.
const { clipboard, downloads, TEMPLATES_SNAPSHOT, templatesSnapshotRef } = vi.hoisted(() => {
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
  const invocationTemplate = (type: string, title: string, inputs: Record<string, ReturnType<typeof fieldInput>>) => ({
    category: 'test',
    classification: 'stable',
    description: '',
    inputs,
    nodePack: 'invokeai',
    outputs: {},
    outputType: `${type}_output`,
    tags: [],
    title,
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
      denoise_latents: invocationTemplate('denoise_latents', 'Denoise Latents', {
        cfg_scale: fieldInput('cfg_scale', 7),
        seed: fieldInput('seed', 0),
        steps: fieldInput('steps', 30),
      }),
      integer: invocationTemplate('integer', 'Integer Primitive', { value: fieldInput('value', 0) }),
      l2i: invocationTemplate('l2i', 'Latents to Image', {}),
      sdxl_compel: invocationTemplate('sdxl_compel', 'SDXL Positive Conditioning', {}),
    },
  };

  return {
    clipboard: { canCopyText: vi.fn(() => true), copyText: vi.fn(() => Promise.resolve()) },
    downloads: { downloadBlob: vi.fn(), downloadText: vi.fn() },
    TEMPLATES_SNAPSHOT: templatesSnapshot,
    templatesSnapshotRef: { current: templatesSnapshot, listeners: new Set<() => void>() },
  };
});

vi.mock('@platform/browser/downloadBlob', () => downloads);
vi.mock('@platform/browser/clipboard', () => clipboard);

// The node card's model markers read the catalog; this suite's graphs need no model.
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  ensureStartersLoaded: vi.fn(),
  useActiveInstallSources: () => new Set<string>(),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: [] }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) => selector({ response: { starter_models: [] } }),
}));

// A store like production's, so the menu's disabled state follows the templates loading.
vi.mock('@features/workflow/react', async (importOriginal) => {
  const { useSyncExternalStore } = await import('react');
  const subscribe = (listener: () => void) => {
    templatesSnapshotRef.listeners.add(listener);

    return () => templatesSnapshotRef.listeners.delete(listener);
  };
  const getSnapshot = () => templatesSnapshotRef.current;

  return {
    ...(await importOriginal<Record<string, unknown>>()),
    getInvocationTemplatesSnapshot: getSnapshot,
    useInvocationTemplatesSnapshot: () => useSyncExternalStore(subscribe, getSnapshot),
  };
});

const setTemplatesSnapshot = (snapshot: InvocationTemplatesSnapshot) =>
  act(() => {
    templatesSnapshotRef.current = snapshot;
    templatesSnapshotRef.listeners.forEach((listener) => listener());
  });

const identity = vi.hoisted(() => ({
  session: { multiuserEnabled: false, phase: 'ready', user: null } as {
    multiuserEnabled: boolean;
    phase: 'ready';
    user: { user_id: string } | null;
  },
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useAuthSession: () => identity.session,
}));

const { createLibraryWorkflowRecordMock, setLibraryWorkflowPublicMock } = vi.hoisted(() => ({
  createLibraryWorkflowRecordMock: vi.fn(),
  setLibraryWorkflowPublicMock: vi.fn(),
}));

// The publication controller reaches the transport through the API module itself, not the queries barrel.
vi.mock('@features/workflow/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  createLibraryWorkflowRecord: createLibraryWorkflowRecordMock,
  setLibraryWorkflowPublic: setLibraryWorkflowPublicMock,
}));

// Supply rendered English strings without starting the HTTP-backed i18n client.
const TRANSLATIONS: Record<string, string> = {
  'common.close': 'Close',
  'graphPreview.aboutThisGraph': 'About this graph',
  'graphPreview.cannotInvoke': "Can't invoke yet",
  'graphPreview.editAsWorkflow': 'Edit as workflow',
  'graphPreview.readyToInvoke': 'Ready to invoke',
  'workflowLibrary.preview.graph': 'Graph',
  'workflowLibrary.preview.graphJson': 'Graph JSON',
  'workflowLibrary.preview.nodes': 'Nodes',
  'graphPreview.copyFailed': 'Failed to copy JSON',
  'graphPreview.copyGraphJson': 'Copy graph JSON',
  'graphPreview.destination': 'Destination',
  'graphPreview.downloadGraphJson': 'Download graph JSON',
  'graphPreview.editInEditor': 'Edit in workflow editor',
  'graphPreview.editInEditorFailed': 'No editable nodes in this graph.',
  'graphPreview.forkIntoProject': 'Fork into new project',
  'graphPreview.forkIntoProjectFailed': 'No nodes to fork into a project.',
  'graphPreview.graphJsonCopied': 'Graph JSON copied',
  'graphPreview.invalidTitle': "This graph can't compile yet.",
  'graphPreview.liveHint': 'Updates as you change settings.',
  'graphPreview.more': 'More',
  'graphPreview.noCompiledGraph': 'No compiled graph is available for "{{graphId}}" yet.',
  'graphPreview.nodeSummary.denoise': '{{count}} steps · cfg {{cfg}}',
  'graphPreview.openedFromPreview': 'Opened from graph preview',
  'graphPreview.saveToLibrary': 'Save to workflow library',
  'graphPreview.saveToLibraryFailed': 'No saveable nodes in this graph.',
  'graphPreview.showNode': 'Show node',
  'workflowLibrary.preview.inputs': 'Inputs',
  'workflowLibrary.preview.nodePackMissing': 'Node pack missing',
  'workflowLibrary.preview.valueNumber': '{{value}}',
  'workflowLibrary.preview.youSetThis': 'You set this',
  'workflowLibrary.publicVisibilityHelp': 'Visible to users of this InvokeAI instance.',
  'workflowLibrary.publicVisibilityPartial':
    'The template was saved, but its public visibility could not be confirmed ({{message}}). Retry to make it public.',
  'workflowLibrary.publicWorkflow': 'Public workflow',
  'workflowLibrary.retry': 'Retry',
  'workflowLibrary.saveFailed': 'Failed to save workflow',
  'workflowLibrary.saveToLibraryConfirm': 'Save',
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
  backendGraph: { edges: [], id: 'backend-graph', nodes: { denoise_latents: { type: 'denoise_latents' } } },
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

/** Generate's shape: compiled, live, with facts, a notice, a value that is not what runs and fields it set. */
const FIXTURE_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  getProvenance: (nodeId, fieldName) => {
    if (nodeId === 'denoise_latents' && fieldName === 'steps') {
      return { label: 'Generate → Steps' };
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

/** The same source recompiled: `steps` changed, or the decoder gone. */
const withGraph = (graph: WorkflowPreviewGraph): GraphPreviewSourceState => ({ ...FIXTURE_SOURCE, graph });
const withSteps = (steps: number) =>
  withGraph({
    ...FIXTURE_GRAPH,
    nodes: FIXTURE_GRAPH.nodes.map((node) =>
      node.id === 'denoise_latents' ? { ...node, inputs: { ...node.inputs, steps } } : node
    ),
  });

/** A widget's last graph: nothing set, nothing live, routed nowhere. */
const PLAIN_SOURCE: GraphPreviewSourceState = {
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

// An unknown-only graph still draws; it converts to an empty document, so document-opening actions must stop.
const UNKNOWN_NODE_GRAPH: WorkflowPreviewGraph = {
  id: 'unknown-node-graph',
  nodes: [{ id: 'mystery', type: 'unknown_type', inputs: { strength: 3 } }],
  edges: [],
  version: 1,
};

const UNKNOWN_NODE_SOURCE: GraphPreviewSourceState = { ...PLAIN_SOURCE, graph: UNKNOWN_NODE_GRAPH };

const invocation = (id: string, type: string, inputs: Record<string, unknown>, label = ''): WorkflowInvocationNode => ({
  data: {
    inputs: Object.fromEntries(Object.entries(inputs).map(([name, value]) => [name, { label: '', name, value }])),
    isIntermediate: true,
    isOpen: true,
    label,
    nodePack: 'invokeai',
    notes: '',
    type,
    useCache: true,
    version: '1.0.0',
  },
  id,
  position: { x: 0, y: 0 },
  type: 'invocation',
});

/** The workflow surface's shape: the document the graph compiles from, its form exposing the steps. */
const DOCUMENT: ProjectGraphState = {
  ...createProjectGraph('workflow-doc', 'Portrait workflow'),
  form: {
    elements: {
      root: { data: { children: ['steps-field'], layout: 'column' }, id: 'root', type: 'container' },
      'steps-field': {
        data: {
          fieldIdentifier: { fieldName: 'steps', nodeId: 'denoise_latents' },
          showDescription: false,
          showShuffle: false,
        },
        id: 'steps-field',
        parentId: 'root',
        type: 'node-field',
      },
    },
    rootElementId: 'root',
  },
  nodes: [invocation('denoise_latents', 'denoise_latents', { cfg_scale: 4, steps: 28 }, 'Sampler')],
};

const DOCUMENT_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  document: DOCUMENT,
  graph: FIXTURE_GRAPH,
  invalidReasons: [],
  isLive: true,
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
  openDocumentInNewProject: vi.fn(),
  openWorkflowEditor: vi.fn(),
});

/** A host's Invoke: the dialog places it and says whether it can run. */
const hostInvoke = (isReady: boolean): GraphPreviewInvoke => ({
  control: <button data-host-invoke="">Host Invoke</button>,
  isReady,
});
const READY = hostInvoke(true);

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
    invoke: GraphPreviewInvoke | null
  ) => (
    <StrictMode>
      <ChakraProvider value={system}>
        <WorkflowUiProvider adapter={workflowUiAdapter}>
          <WorkflowGraphPreviewProvider adapter={graphPreviewPort}>
            <GraphPreviewDialog
              graphId="preview-graph-id"
              invoke={invoke ?? undefined}
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
    invoke: GraphPreviewInvoke | null = READY
  ) => {
    await act(() => {
      root.render(dialogTree(source, isOpen, sourceId, invoke));
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
    clipboard.copyText.mockClear();
    downloads.downloadBlob.mockReset();
    downloads.downloadText.mockReset();
    createLibraryWorkflowRecordMock.mockReset();
    setLibraryWorkflowPublicMock.mockReset();
    identity.session = { multiuserEnabled: false, phase: 'ready', user: null };
    templatesSnapshotRef.current = TEMPLATES_SNAPSHOT;
  });

  afterEach(async () => {
    await act(() => root.unmount());
    // The More menu's content portals to `document.body`, outside the React root this suite unmounts above; sweep it
    // so a leftover node from one test can't answer a `[role="menuitem"]` query in the next.
    document.querySelectorAll('[data-scope="menu"]').forEach((element) => element.remove());
    host.remove();
  });

  const resizeViewport = async (width: number, height: number) => {
    const previous = { height: window.innerHeight, width: window.innerWidth };

    await page.viewport(width, height);
    onTestFinished(() => page.viewport(previous.width, previous.height));
  };
  /** The dialog's opening animation scales it; measure once it has played. */
  const settleOpening = () =>
    act(async () => {
      await vi.waitFor(() => {
        const content = document.querySelector('[data-scope="dialog"][data-part="content"]')!;

        expect(
          content.getAnimations({ subtree: true }).filter((animation) => animation.playState === 'running')
        ).toEqual([]);
      });
    });
  const stage = () => document.querySelector<HTMLElement>('[data-graph-preview-stage]');
  const rail = () => document.querySelector<HTMLElement>('[data-graph-preview-rail]');
  const title = () => document.querySelector('[data-scope="dialog"][data-part="title"]')?.textContent ?? null;
  const description = () =>
    document.querySelector('[data-scope="dialog"][data-part="description"]')?.textContent ?? null;
  const previewNode = (id: string) => stage()?.querySelector<HTMLElement>(`.react-flow__node[data-id="${id}"]`) ?? null;
  const drawnNodeIds = () =>
    [...(stage()?.querySelectorAll<HTMLElement>('.react-flow__node') ?? [])].map((node) => node.dataset.id);
  const nodeCard = () => stage()?.querySelector<HTMLElement>('[data-preview-node-card]') ?? null;
  const press = (key: string) => act(() => userEvent.keyboard(key));

  const waitForGraph = () =>
    act(async () => {
      await vi.waitFor(() => expect(stage()?.querySelector('.react-flow__node')).not.toBeNull());
    });

  const openCard = async (nodeId: string) => {
    await act(() => userEvent.click(previewNode(nodeId)!));
    await vi.waitFor(() => expect(nodeCard()?.dataset.previewNodeCard).toBe(nodeId));
  };

  const clickButtonWithText = async (text: string, scope: ParentNode = document) => {
    const button = [...scope.querySelectorAll('button')].find((candidate) =>
      (candidate.textContent ?? '').includes(text)
    );
    expect(button).not.toBeUndefined();

    await act(() => {
      button?.click();
    });
  };

  // Wait for lazy menu portal mounting before querying items.
  const openMoreMenu = async () => {
    await act(() => document.querySelector<HTMLButtonElement>('[data-preview-more]')?.click());
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

    // Zag selects only the highlighted item, so this takes a real pointer click, which hovers it first.
    await act(() => userEvent.click(item as HTMLElement));
  };

  // Await the detached save handler's async chain before asserting serialization, creation, and notification.
  const flushAsync = () =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });

  const deferred = <T,>() => {
    let reject!: (reason?: unknown) => void;
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
      resolve = resolvePromise;
      reject = rejectPromise;
    });

    return { promise, reject, resolve };
  };

  it('titles itself with the source and says when it follows the settings', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect(title()).toBe('Generate');
    expect(description()).toBe('Updates as you change settings.');

    await renderDialog(PLAIN_SOURCE);

    expect(description()).toBeNull();
  });

  it('names a document source by its workflow', async () => {
    await renderDialog(DOCUMENT_SOURCE, true, 'workflow');

    expect(title()).toBe('Portrait workflow');
  });

  it('draws a compiled graph as the library does: helpers folded, set fields and the result marked', async () => {
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();

    // The seed primitive folds into a pill on the denoiser, showing what runs rather than the compiled placeholder.
    expect(drawnNodeIds()).toEqual(['denoise_latents', 'l2i']);
    expect(previewNode('denoise_latents')?.querySelector('[data-preview-pill="seed"]')?.textContent?.trim()).toBe(
      'Integer Primitive regenerated each run'
    );
    // The steps came from a setting: the node is marked as set; the decoder is what the graph makes.
    expect(previewNode('denoise_latents')?.querySelector('[data-marker="exposed"]')).not.toBeNull();
    expect(previewNode('l2i')?.querySelector('[data-marker="result"]')).not.toBeNull();
    expect(stage()?.querySelector('[data-preview-legend]')?.textContent).toContain('You set this');
  });

  it('draws nodes this server has no template for', async () => {
    await renderDialog(UNKNOWN_NODE_SOURCE);
    await waitForGraph();

    expect(drawnNodeIds()).toEqual(['mystery']);
    expect(previewNode('mystery')?.textContent).toContain('Node pack missing');

    await openCard('mystery');

    expect(nodeCard()?.textContent).toContain('strength');
  });

  it('lists the source’s notices beside the graph, and a notice shows its node', async () => {
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();

    expect(rail()?.querySelector('[data-graph-preview-notice]')?.textContent).toContain('Seed is randomized');

    await clickButtonWithText('Show node', rail()!);

    // The seed is folded in the simplified view, so the graph opens in full with the seed's card.
    await vi.waitFor(() => expect(nodeCard()?.dataset.previewNodeCard).toBe('seed'));
    expect(drawnNodeIds()).toEqual(['seed', 'denoise_latents', 'l2i']);
    expect(nodeCard()?.textContent).toContain('regenerated each run');

    // Asked again after the card closed, it shows the node again.
    await press('{Escape}');
    expect(nodeCard()).toBeNull();
    await clickButtonWithText('Show node', rail()!);
    await vi.waitFor(() => expect(nodeCard()?.dataset.previewNodeCard).toBe('seed'));
  });

  it('links a set field to its setting, which it focuses before closing', async () => {
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();
    await openCard('denoise_latents');

    const links = [...nodeCard()!.querySelectorAll('[data-preview-field-source]')].map((link) => link.textContent);
    expect(links).toEqual(['Generate → Steps']);
    // Named as the setting that set it.
    expect([...nodeCard()!.querySelectorAll('dt')].some((label) => label.textContent?.startsWith('Steps'))).toBe(true);

    await clickButtonWithText('Generate → Steps', nodeCard()!);

    expect(graphPreviewPort.focusSource).toHaveBeenCalledWith('generate');
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('keeps an open card through live recompiles while its node survives, and closes it once the node is gone', async () => {
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();
    await openCard('denoise_latents');
    expect(nodeCard()?.textContent).toContain('28');

    // A value changes: the same drawing, the card showing the new value.
    await renderDialog(withSteps(12));

    expect(nodeCard()?.dataset.previewNodeCard).toBe('denoise_latents');
    expect(nodeCard()?.textContent).toContain('12');
    expect(nodeCard()?.textContent).not.toContain('28');

    // The structure changes (the decoder goes): drawn afresh, the card still open on its node.
    await renderDialog(
      withGraph({ ...FIXTURE_GRAPH, edges: FIXTURE_GRAPH.edges.slice(0, 1), nodes: FIXTURE_GRAPH.nodes.slice(0, 2) })
    );
    await vi.waitFor(() => expect(drawnNodeIds()).not.toContain('l2i'));
    await vi.waitFor(() => expect(nodeCard()?.dataset.previewNodeCard).toBe('denoise_latents'));

    await renderDialog(
      withGraph({
        ...FIXTURE_GRAPH,
        edges: [],
        nodes: FIXTURE_GRAPH.nodes.filter((node) => node.id !== 'denoise_latents'),
      })
    );
    await waitForGraph();
    await vi.waitFor(() => expect(drawnNodeIds()).not.toContain('denoise_latents'));

    expect(nodeCard()).toBeNull();
  });

  it('closes an open card on Escape before the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();
    await openCard('denoise_latents');

    await press('{Escape}');

    expect(nodeCard()).toBeNull();
    expect(onOpenChange).not.toHaveBeenCalled();

    await press('{Escape}');

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  const tabs = () => [...(stage()?.querySelectorAll<HTMLElement>('[role="tab"]') ?? [])];
  const selectTab = async (view: 'nodes' | 'inputs' | 'graph') => {
    const tab = tabs().find((candidate) => candidate.id.endsWith(`-tab-${view}`));

    expect(tab, `no ${view} tab`).not.toBeUndefined();
    await act(() => tab!.click());
  };
  const moreButton = () => document.querySelector<HTMLElement>('[data-preview-more]');

  it('offers Nodes and Graph for a compiled source, and Inputs too for a document, whose form it reads', async () => {
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();

    expect(tabs().map((tab) => tab.textContent)).toEqual(['Nodes', 'Graph']);

    await renderDialog(DOCUMENT_SOURCE, true, 'workflow');
    await waitForGraph();

    // Drawn from the document: the author's label names the node.
    expect(previewNode('denoise_latents')?.textContent).toContain('Sampler');
    expect(tabs().map((tab) => tab.textContent)).toEqual(['Nodes', 'Inputs 1', 'Graph']);

    await selectTab('inputs');

    expect(stage()?.querySelector('[data-preview-input="steps-field"]')?.textContent).toContain('steps');
  });

  it('keeps the tabs and More where they are in every view; only find comes and goes', async () => {
    await resizeViewport(1288, 942);
    await renderDialog(DOCUMENT_SOURCE, true, 'workflow');
    await waitForGraph();
    await settleOpening();

    const tabStrip = () => stage()!.querySelector<HTMLElement>('[role="tablist"]')!.getBoundingClientRect().toJSON();
    const more = () => moreButton()!.getBoundingClientRect().toJSON();
    const nodes = { more: more(), tabs: tabStrip() };

    expect(stage()?.querySelector('[data-preview-find]')).not.toBeNull();

    for (const view of ['inputs', 'graph', 'nodes'] as const) {
      await selectTab(view);
      await vi.waitFor(() => expect({ more: more(), tabs: tabStrip() }).toEqual(nodes));
      expect(stage()?.querySelector('[data-preview-find]') !== null).toBe(view === 'nodes');
    }
  });

  it('shows the compiled backend graph in the Graph view, serialized only once that view shows', async () => {
    const backendGraph = FIXTURE_GRAPH.backendGraph as Record<string, unknown>;
    const toJSON = vi.fn(() => backendGraph);

    await renderDialog(withGraph({ ...FIXTURE_GRAPH, backendGraph: { toJSON } }));
    await waitForGraph();

    expect(toJSON).not.toHaveBeenCalled();

    await selectTab('graph');

    await vi.waitFor(() =>
      expect(stage()?.querySelector('[data-preview-json] code')?.textContent).toBe(
        JSON.stringify(backendGraph, null, 2)
      )
    );
  });

  it('shows why a compiled source cannot compile, in the Nodes and Graph views alike', async () => {
    await renderDialog(INVALID_SOURCE);

    const reason = "This graph can't compile yet. Height must be a multiple of 8.";

    expect(stage()?.querySelector('[role="tabpanel"] [role="alert"]')?.textContent).toBe(reason);

    await selectTab('graph');

    expect(stage()?.querySelector('[role="tabpanel"] [role="alert"]')?.textContent).toBe(reason);
    // The menu stays where it is and says why there is no JSON to export.
    await openMoreMenu();
    expect(findMenuItemWithText('Copy graph JSON')?.textContent).toContain("This graph can't compile yet.");
  });

  it('draws a document that does not compile, with the reason beside it', async () => {
    await renderDialog(
      { ...DOCUMENT_SOURCE, graph: null, invalidReasons: ['Height must be a multiple of 8.'] },
      true,
      'workflow'
    );
    await waitForGraph();

    expect(drawnNodeIds()).toEqual(['denoise_latents']);
    expect(rail()?.querySelector('[data-graph-preview-blocked]')?.textContent).toBe(
      "This graph can't compile yet. Height must be a multiple of 8."
    );
  });

  it('says when a widget has no compiled graph yet', async () => {
    await renderDialog({ ...PLAIN_SOURCE, graph: null });

    expect(stage()?.querySelector('[role="tabpanel"]')?.textContent).toBe(
      'No compiled graph is available for "preview-graph-id" yet.'
    );
  });

  it('opens at the library dialog’s size, so the two previews read the same', async () => {
    await resizeViewport(1288, 942);
    await renderDialog(FIXTURE_SOURCE);
    await settleOpening();

    const box = document.querySelector('[data-scope="dialog"][data-part="content"]')!.getBoundingClientRect();

    // The window less 4rem wide (under the 80rem cap) and 80% of its height.
    expect(Math.round(box.width)).toBe(1288 - 4 * 16);
    expect(Math.round(box.height)).toBe(Math.round(942 * 0.8));
  });

  it('fits a graph clear of the stage’s controls, on opening and on Fit', async () => {
    await resizeViewport(1288, 942);
    // Two results at opposite corners, spanning about the stage's shape: fitted to the whole stage, the top-left one
    // would sit under the legend.
    const corners: WorkflowPreviewGraph = {
      edges: [],
      id: 'corners',
      nodes: [
        { id: 'top-left', inputs: {}, type: 'l2i' },
        { id: 'bottom-right', inputs: {}, type: 'l2i' },
      ],
      version: 1,
    };

    await renderDialog({
      ...PLAIN_SOURCE,
      graph: corners,
      positionHints: { 'bottom-right': { x: 1890, y: 1038 }, 'top-left': { x: 0, y: 0 } },
    });
    await waitForGraph();

    const expectClear = (selector: string) => {
      const controls = [...stage()!.querySelectorAll(selector)].map((control) => control.getBoundingClientRect());

      for (const node of stage()!.querySelectorAll('.react-flow__node')) {
        const box = node.getBoundingClientRect();

        for (const control of controls) {
          const overlaps =
            box.left < control.right &&
            control.left < box.right &&
            box.top < control.bottom &&
            control.top < box.bottom;

          expect(overlaps, `${(node as HTMLElement).dataset.id} under a control`).toBe(false);
        }
      }
    };

    await settleOpening();
    // Opening: fitted or, when too tall to read fitted, from the top; either way the legend covers no node.
    expect(stage()?.querySelector('[data-preview-legend]')).not.toBeNull();
    expectClear('[data-preview-legend]');

    await clickButtonWithText('workflowLibrary.preview.fit', stage()!);
    // Fitted: the whole graph in the stage's open area, clear of every control.
    await vi.waitFor(() => expectClear('[data-preview-overlay]'));
  });

  it('exposes the preview controls and drawn nodes with accessible names', async () => {
    const wasDark = document.documentElement.classList.contains('dark');
    document.documentElement.classList.add('dark');
    onTestFinished(() => {
      if (!wasDark) {
        document.documentElement.classList.remove('dark');
      }
    });
    await renderDialog(PLAIN_SOURCE);
    await waitForGraph();
    await settleOpening();
    const dialog = document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]')!;
    expect(await auditAccessibility(dialog, { rules: { 'label-content-name-mismatch': { enabled: true } } })).toEqual(
      []
    );
  });

  it('widens every chip for its longest title where the stage has the room', async () => {
    // Keep the title between the minimum and maximum chip widths, independent of the host's fonts and root size.
    const typography = document.createElement('style');
    typography.textContent = ':root { font-size: 16px !important; --chakra-fonts-body: monospace; }';
    document.head.append(typography);
    onTestFinished(() => typography.remove());
    templatesSnapshotRef.current = {
      ...TEMPLATES_SNAPSHOT,
      templates: {
        ...TEMPLATES_SNAPSHOT.templates,
        sdxl_compel: { ...TEMPLATES_SNAPSHOT.templates.sdxl_compel!, title: 'Positive Conditioning' },
      },
    };
    await resizeViewport(1288, 942);
    const graph: WorkflowPreviewGraph = {
      edges: [
        {
          id: 'e',
          sourceField: 'conditioning',
          sourceNodeId: 'prompt',
          targetField: 'latents',
          targetNodeId: 'decode',
        },
      ],
      id: 'wide',
      nodes: [
        { id: 'prompt', inputs: {}, type: 'sdxl_compel' },
        { id: 'decode', inputs: {}, type: 'l2i' },
      ],
      version: 1,
    };

    await renderDialog({ ...PLAIN_SOURCE, graph });
    await waitForGraph();

    const width = (id: string) => previewNode(id)!.offsetWidth;
    const title = previewNode('prompt')!.querySelector<HTMLElement>('[data-preview-chip] p')!;

    // One width for every chip, past the narrowest, and the long title whole.
    expect(width('prompt')).toBeGreaterThan(136);
    expect(width('decode')).toBe(width('prompt'));
    expect(title.scrollWidth).toBeLessThanOrEqual(title.clientWidth);
  });

  it('keeps its drawing when a host hands a new source object around the same graph each render', async () => {
    // A host that rebuilds the source on every render, its callbacks fresh each time.
    const asked: string[] = [];
    const rebuilt = () => ({
      ...FIXTURE_SOURCE,
      getProvenance: (nodeId: string, fieldName: string) => {
        asked.push(nodeId);
        return FIXTURE_SOURCE.getProvenance!(nodeId, fieldName);
      },
    });

    await renderDialog(rebuilt());
    await waitForGraph();
    await openCard('denoise_latents');
    const chip = previewNode('l2i');
    asked.length = 0;

    for (let render = 0; render < 3; render += 1) {
      await renderDialog(rebuilt());
    }

    // Nothing rebuilt: the seed's fields (no card shows them) were not asked about again; the drawing and card stay.
    expect(asked).not.toContain('seed');
    expect(previewNode('l2i')).toBe(chip);
    expect(nodeCard()?.dataset.previewNodeCard).toBe('denoise_latents');
  });

  it('starts no model or starter request, open or closed', async () => {
    const { ensureModelsLoaded, ensureStartersLoaded } = await import('@features/models');

    vi.mocked(ensureModelsLoaded).mockClear();
    vi.mocked(ensureStartersLoaded).mockClear();
    await renderDialog(FIXTURE_SOURCE, false);
    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();

    expect(ensureModelsLoaded).not.toHaveBeenCalled();
    expect(ensureStartersLoaded).not.toHaveBeenCalled();
  });

  it('reports when its close transition has finished, so hosts can drop the mount', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect(onExitComplete).not.toHaveBeenCalled();

    await renderDialog(FIXTURE_SOURCE, false);

    // The exit animation runs in real time; polling inside `act` keeps its final commit inside an open act scope.
    await act(async () => {
      await vi.waitFor(() => {
        expect(onExitComplete).toHaveBeenCalled();
      });
    });
  });

  it('reopens on the nodes, unfiltered, with no card open', async () => {
    const findField = () => document.querySelector<HTMLInputElement>('[data-preview-find]');

    await renderDialog(FIXTURE_SOURCE);
    await waitForGraph();
    await act(() => userEvent.fill(findField()!, 'latents'));
    await openCard('denoise_latents');

    await renderDialog(FIXTURE_SOURCE, false);
    await act(async () => {
      await vi.waitFor(() => expect(onExitComplete).toHaveBeenCalled());
    });
    await renderDialog(FIXTURE_SOURCE, true);
    await waitForGraph();

    expect(nodeCard()).toBeNull();
    expect(findField()?.value).toBe('');
  });

  it('More → Copy graph JSON copies the compiled backend graph', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Copy graph JSON');
    await flushAsync();

    expect(clipboard.copyText).toHaveBeenCalledWith(JSON.stringify(FIXTURE_GRAPH.backendGraph, null, 2));
    expect(workflowUiAdapter.notifications.success).toHaveBeenCalledWith('Graph JSON copied');
  });

  it('More → Download graph JSON downloads the compiled backend graph', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Download graph JSON');

    expect(downloads.downloadText).toHaveBeenCalledWith(
      JSON.stringify(FIXTURE_GRAPH.backendGraph, null, 2),
      'graph.json',
      'application/json'
    );
  });

  it('More → Edit in workflow editor adds the document as a project workflow, opens the editor, closes the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Edit in workflow editor');

    expect(workflowUiAdapter.commands.addWorkflow).toHaveBeenCalledTimes(1);
    const [document_, options] = vi.mocked(workflowUiAdapter.commands.addWorkflow).mock.calls[0] ?? [];
    expect(document_?.nodes).toHaveLength(3);
    // A blank starter workflow may be taken over; an edited one gets the preview beside it.
    expect(options).toEqual({ label: 'Opened from graph preview', reusePlaceholder: true });
    expect(graphPreviewPort.openWorkflowEditor).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('More → Fork into new project hands the named document to the port and closes the dialog', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
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

  it('More → Fork into new project copies a document source, keeping its labels and form', async () => {
    await renderDialog(
      { ...DOCUMENT_SOURCE, graph: null, invalidReasons: ['Height must be a multiple of 8.'] },
      true,
      'workflow'
    );

    await openMoreMenu();
    await clickMenuItemWithText('Fork into new project');

    const [document_] = vi.mocked(graphPreviewPort.openDocumentInNewProject).mock.calls[0] ?? [];
    // A copy of its own, even while it does not compile.
    expect(document_?.id).not.toBe(DOCUMENT.id);
    expect(document_?.name).toBe('Portrait workflow');
    expect(document_?.nodes.map((node) => (node.type === 'invocation' ? node.data.label : null))).toEqual(['Sampler']);
    expect(document_?.form).toEqual(DOCUMENT.form);
  });

  it('hides Edit in workflow editor for the workflow source', async () => {
    await renderDialog(FIXTURE_SOURCE, true, 'workflow');

    await openMoreMenu();

    expect(findMenuItemWithText('Edit in workflow editor')).toBeUndefined();
    expect(findMenuItemWithText('Save to workflow library')).not.toBeUndefined();
    expect(findMenuItemWithText('Download graph JSON')).not.toBeUndefined();
  });

  const footer = () => document.querySelector<HTMLElement>('[data-graph-preview-footer]');
  const readiness = () => footer()?.querySelector<HTMLElement>('[data-graph-preview-readiness]') ?? null;
  const railFacts = () =>
    [...(rail()?.querySelectorAll('[data-graph-preview-facts] dt') ?? [])].map(
      (label) => `${label.textContent}: ${label.nextElementSibling?.textContent}`
    );

  it('says only whether it can invoke in the footer, the run’s facts and destination in the rail', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect(readiness()?.textContent).toBe('Ready to invoke');
    expect(readiness()?.querySelector('[aria-live="polite"]')?.textContent).toBe('Ready to invoke');
    expect(footer()?.querySelector('[data-host-invoke]')).not.toBeNull();
    // Above the notices, as label and value; the destination said there, not again under the title.
    expect(railFacts()).toEqual(['Steps: 28', 'Model: SDXL', 'Destination: Gallery']);
    expect(
      rail()!
        .querySelector('[data-graph-preview-facts]')!
        .compareDocumentPosition(rail()!.querySelector('[data-graph-preview-notice]')!) &
        Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
    expect(description()).toBe('Updates as you change settings.');

    await renderDialog(FIXTURE_SOURCE, true, 'generate', hostInvoke(false));

    expect(readiness()?.textContent).toBe("Can't invoke yet");
    expect(railFacts()).toEqual(['Steps: 28', 'Model: SDXL', 'Destination: Gallery']);
  });

  it('shows the rail for facts alone, and none for a source with nothing to say beside the graph', async () => {
    await renderDialog({ ...PLAIN_SOURCE, destinationLabel: 'Gallery' });

    expect(railFacts()).toEqual(['Destination: Gallery']);

    await renderDialog(PLAIN_SOURCE);

    expect(rail()).toBeNull();
  });

  it('offers a compiled source as a workflow to edit, left of Invoke', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect([...footer()!.querySelectorAll('button')].map((button) => button.textContent)).toEqual([
      'Edit as workflow',
      'Host Invoke',
    ]);
    await clickButtonWithText('Edit as workflow', footer()!);

    expect(workflowUiAdapter.commands.addWorkflow).toHaveBeenCalledTimes(1);
    expect(graphPreviewPort.openWorkflowEditor).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);

    // A document is already a workflow.
    await renderDialog(DOCUMENT_SOURCE, true, 'workflow');

    expect(footer()?.querySelector('[data-graph-preview-edit-as-workflow]')).toBeNull();
  });

  it('closes from the header, with no footer Close', async () => {
    await renderDialog(FIXTURE_SOURCE);

    const close = document.querySelector<HTMLButtonElement>('[data-scope="dialog"][data-part="close-trigger"]');

    // In the header, beside the title.
    expect(close?.parentElement?.contains(document.querySelector('[data-scope="dialog"][data-part="title"]'))).toBe(
      true
    );
    expect(footer()?.textContent).not.toContain('Close');
    await act(() => close!.click());

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('without a host Invoke has no footer; the facts stay in the rail', async () => {
    await renderDialog(FIXTURE_SOURCE, true, 'generate', null);

    expect(footer()).toBeNull();
    expect(railFacts()).toEqual(['Steps: 28', 'Model: SDXL', 'Destination: Gallery']);
    expect(document.querySelector('[data-preview-more]')).not.toBeNull();
  });

  it('More → Save to workflow library names the document from the source label and notifies success', async () => {
    createLibraryWorkflowRecordMock.mockImplementation((workflow: Record<string, unknown>) =>
      Promise.resolve({ name: workflow.name, revision: 1, workflow, workflow_id: 'library-workflow-1' })
    );

    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(1);
    const [serialized, options] = createLibraryWorkflowRecordMock.mock.calls[0] ?? [];
    expect(serialized).toMatchObject({ name: 'Generate' });
    // A client-reserved id makes a resend after a lost answer idempotent.
    expect(options).toMatchObject({ reservedId: expect.any(String) });
    expect(options).not.toHaveProperty('isPublic');
    expect(workflowUiAdapter.notifications.success).toHaveBeenCalledWith(
      'Workflow saved',
      'Saved "Generate" to the library.'
    );
    // A preview is not a project workflow, so nothing in the project is pointed at the new template.
    expect(workflowUiAdapter.commands.setWorkflowSource).not.toHaveBeenCalled();
    expect(setLibraryWorkflowPublicMock).not.toHaveBeenCalled();
  });

  it('More → Save to workflow library offers the public visibility choice in multiuser mode', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    createLibraryWorkflowRecordMock.mockImplementation(
      (workflow: Record<string, unknown>, options: { reservedId?: string }) =>
        Promise.resolve({
          name: String(workflow.name),
          revision: 1,
          workflow,
          workflow_id: options.reservedId,
          is_public: false,
        })
    );
    setLibraryWorkflowPublicMock.mockImplementation((workflowId: string) =>
      Promise.resolve({ name: 'Generate', revision: 1, workflow: {}, workflow_id: workflowId, is_public: true })
    );

    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');

    const checkbox = document.querySelector<HTMLInputElement>('input[type="checkbox"][name="isPublic"]');
    expect(checkbox).not.toBeNull();
    expect(checkbox?.checked).toBe(false);
    await act(() => checkbox?.click());
    await clickButtonWithText('Save');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(1);
    const [, options] = createLibraryWorkflowRecordMock.mock.calls[0] ?? [];
    expect(options).toMatchObject({ reservedId: expect.any(String) });
    expect(options).not.toHaveProperty('isPublic');
    expect(setLibraryWorkflowPublicMock).toHaveBeenCalledWith(
      (options as { reservedId: string }).reservedId,
      true,
      expect.anything()
    );
    expect(workflowUiAdapter.commands.setWorkflowSource).not.toHaveBeenCalled();
  });

  it('retries a failed visibility change with the same preview template and public choice', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    createLibraryWorkflowRecordMock.mockImplementation(
      (workflow: Record<string, unknown>, options: { reservedId?: string }) =>
        Promise.resolve({
          name: String(workflow.name),
          revision: 1,
          workflow,
          workflow_id: options.reservedId,
          is_public: false,
        })
    );
    setLibraryWorkflowPublicMock
      .mockRejectedValueOnce(new Error('visibility unavailable'))
      .mockImplementationOnce((workflowId: string) =>
        Promise.resolve({ name: 'Generate', revision: 1, workflow: {}, workflow_id: workflowId, is_public: true })
      );

    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    const checkbox = document.querySelector<HTMLInputElement>('input[type="checkbox"][name="isPublic"]');
    expect(checkbox).not.toBeNull();
    await act(() => checkbox?.click());
    await clickButtonWithText('Save');
    await flushAsync();

    const createCall = createLibraryWorkflowRecordMock.mock.calls[0];
    expect(createCall).toBeDefined();
    if (!createCall) {
      throw new Error('Expected the preview workflow to be created');
    }
    const reservedId = (createCall[1] as { reservedId: string }).reservedId;
    expect(setLibraryWorkflowPublicMock).toHaveBeenCalledTimes(1);
    expect(setLibraryWorkflowPublicMock).toHaveBeenCalledWith(reservedId, true, expect.anything());
    expect(createCall[1]).not.toHaveProperty('isPublic');
    expect([...document.querySelectorAll('button')].some((button) => button.textContent?.includes('Retry'))).toBe(true);
    expect(document.body.textContent).toContain(
      'The template was saved, but its public visibility could not be confirmed (visibility unavailable). Retry to make it public.'
    );

    await clickButtonWithText('Retry');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(1);
    expect(setLibraryWorkflowPublicMock.mock.calls.map((call) => call.slice(0, 2))).toEqual([
      [reservedId, true],
      [reservedId, true],
    ]);
    expect(workflowUiAdapter.commands.setWorkflowSource).not.toHaveBeenCalled();
  });

  it('More → Save to workflow library retries a lost answer once under the same reserved id, never claiming success', async () => {
    createLibraryWorkflowRecordMock.mockRejectedValue(new Error('network down'));

    await renderDialog(FIXTURE_SOURCE);

    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(2);
    const [, first] = createLibraryWorkflowRecordMock.mock.calls[0] ?? [];
    const [, second] = createLibraryWorkflowRecordMock.mock.calls[1] ?? [];
    expect((second as { reservedId: string }).reservedId).toBe((first as { reservedId: string }).reservedId);
    expect(workflowUiAdapter.notifications.success).not.toHaveBeenCalled();
  });

  it('More → Save to workflow library bails with an error notification when the graph has no saveable nodes', async () => {
    await renderDialog(UNKNOWN_NODE_SOURCE);

    await openMoreMenu();
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
    await openMoreMenu();

    const item = findMenuItemWithText('Save to workflow library');
    expect(item).not.toBeUndefined();
    expect(item?.getAttribute('aria-disabled')).toBe('true');
  });

  it('disables Fork into new project until invocation templates load, then restores it', async () => {
    templatesSnapshotRef.current = { error: null, status: 'loading', templates: {} };

    await renderDialog(FIXTURE_SOURCE);
    await openMoreMenu();

    let item = findMenuItemWithText('Fork into new project');
    expect(item?.getAttribute('aria-disabled')).toBe('true');

    await setTemplatesSnapshot(TEMPLATES_SNAPSHOT);

    item = findMenuItemWithText('Fork into new project');
    expect(item?.getAttribute('aria-disabled')).not.toBe('true');
    await clickMenuItemWithText('Fork into new project');

    expect(graphPreviewPort.openDocumentInNewProject).toHaveBeenCalledTimes(1);
  });

  it('discards a preview visibility failure that settles after the preview unmounts', async () => {
    identity.session = { multiuserEnabled: true, phase: 'ready', user: { user_id: 'user-a' } };
    const visibilityRequest = deferred<Record<string, unknown>>();
    const publicationController = getWorkflowPublicationController();
    const publishSpy = vi.spyOn(publicationController, 'publish');
    createLibraryWorkflowRecordMock.mockImplementation(
      (_workflow: Record<string, unknown>, options: { reservedId?: string }) =>
        Promise.resolve({
          name: 'Generate',
          revision: 1,
          workflow: {},
          workflow_id: options.reservedId,
          is_public: false,
        })
    );
    setLibraryWorkflowPublicMock.mockReturnValue(visibilityRequest.promise);

    await renderDialog(FIXTURE_SOURCE);
    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    const checkbox = document.querySelector<HTMLInputElement>('input[type="checkbox"][name="isPublic"]');
    expect(checkbox).not.toBeNull();
    await act(() => checkbox?.click());
    await clickButtonWithText('Save');
    await flushAsync();

    expect(setLibraryWorkflowPublicMock).toHaveBeenCalledTimes(1);
    const request = publishSpy.mock.calls[0]?.[0];
    expect(request).toBeDefined();
    if (!request) {
      throw new Error('Expected the preview publication request');
    }

    await act(() => root.render(null));
    visibilityRequest.reject(new Error('visibility unavailable'));
    await flushAsync();

    expect(getUnresolvedWorkflowPublication('project-1', request.workflowId)).toBeNull();
    publishSpy.mockRestore();
  });

  it('discards a single-user preview retry failure that settles after the preview unmounts', async () => {
    const retry = deferred<Record<string, unknown>>();
    const publicationController = getWorkflowPublicationController();
    const publishSpy = vi.spyOn(publicationController, 'publish');
    createLibraryWorkflowRecordMock
      .mockRejectedValueOnce(new Error('first attempt lost'))
      .mockReturnValueOnce(retry.promise);

    await renderDialog(FIXTURE_SOURCE);
    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(2);
    const request = publishSpy.mock.calls[0]?.[0];
    publishSpy.mockRestore();
    expect(request).toBeDefined();
    if (!request) {
      throw new Error('Expected the preview publication request');
    }

    await act(() => root.render(null));
    retry.reject(new Error('retry unavailable'));
    await flushAsync();

    expect(getUnresolvedWorkflowPublication('project-1', request.workflowId)).toBeNull();
  });

  it('cleans earlier unresolved single-user saves after a later preview save succeeds', async () => {
    const publicationController = getWorkflowPublicationController();
    const publishSpy = vi.spyOn(publicationController, 'publish');
    createLibraryWorkflowRecordMock
      .mockRejectedValueOnce(new Error('first attempt lost'))
      .mockRejectedValueOnce(new Error('retry unavailable'))
      .mockImplementation((workflow: Record<string, unknown>, options: { reservedId?: string }) =>
        Promise.resolve({
          name: String(workflow.name),
          revision: 1,
          workflow,
          workflow_id: options.reservedId,
          is_public: true,
        })
      );

    await renderDialog(FIXTURE_SOURCE);
    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    const firstRequest = publishSpy.mock.calls[0]?.[0];
    publishSpy.mockRestore();
    expect(firstRequest).toBeDefined();
    if (!firstRequest) {
      throw new Error('Expected the first preview publication request');
    }

    await openMoreMenu();
    await clickMenuItemWithText('Save to workflow library');
    await flushAsync();

    expect(createLibraryWorkflowRecordMock).toHaveBeenCalledTimes(3);
    expect(workflowUiAdapter.notifications.success).toHaveBeenCalledTimes(1);

    await act(() => root.render(null));

    expect(getUnresolvedWorkflowPublication('project-1', firstRequest.workflowId)).toBeNull();
  });
});
