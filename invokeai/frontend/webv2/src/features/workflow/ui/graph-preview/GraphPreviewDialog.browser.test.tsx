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

import { GraphPreviewDialog } from './GraphPreviewDialog';

// xyflow stays out of this shell test — the flow pane itself is covered
// elsewhere (`GraphPreviewFlow`'s own tests).
vi.mock('./GraphPreviewFlow', () => ({
  GraphPreviewFlow: () => <div data-flow-stub />,
  documentToPreviewGraph: () => {
    throw new Error('not used');
  },
}));

// `downloadText` and the invocation templates snapshot both need to exist
// before `vi.mock` factories below run (they're hoisted above the imports
// that would otherwise define them), so they're built through `vi.hoisted`.
const { downloads, TEMPLATES_SNAPSHOT } = vi.hoisted(() => {
  const fieldInput = (name: string, defaultValue: unknown) => ({
    default: defaultValue,
    description: '',
    exclusiveMaximum: null,
    exclusiveMinimum: null,
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

  return {
    downloads: { downloadBlob: vi.fn(), downloadText: vi.fn() },
    TEMPLATES_SNAPSHOT: {
      error: null,
      status: 'loaded' as const,
      templates: {
        denoise_latents: invocationTemplate('denoise_latents', {
          cfg_scale: fieldInput('cfg_scale', 7),
          steps: fieldInput('steps', 30),
        }),
        integer: invocationTemplate('integer', { value: fieldInput('value', 0) }),
        l2i: invocationTemplate('l2i', {}),
      },
    },
  };
});

vi.mock('@platform/browser/downloadBlob', () => downloads);

// The "Open as" menu (`GraphPreviewOpenAsMenu`) reads the invocation
// templates snapshot through the reactive hook, not the plain getter, so its
// disabled state can update live if the menu opens while templates are still
// loading — this stubs that hook (and the getter, for symmetry) with a
// pre-loaded snapshot covering the fixture graph's node types.
vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getInvocationTemplatesSnapshot: () => TEMPLATES_SNAPSHOT,
  useInvocationTemplatesSnapshot: () => TEMPLATES_SNAPSHOT,
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
  'graphPreview.saveToLibraryHint': 'Reusable, leaves this project alone',
  'graphPreview.selectNode': 'Select a node for details.',
  'graphPreview.setBy': 'Set by',
  'graphPreview.showNode': 'show node',
  'graphPreview.thisGraph': 'This graph',
  'graphPreview.title': 'Graph preview',
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

const INVALID_SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  graph: null,
  invalidReasons: ['Height must be a multiple of 8.'],
  isLive: true,
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
  openWorkflowEditor: vi.fn(),
});

describe('GraphPreviewDialog', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: (isOpen: boolean) => void;
  let graphPreviewPort: WorkflowGraphPreviewPort;
  let workflowUiAdapter: WorkflowUiAdapter;

  const renderDialog = async (
    source: GraphPreviewSourceState,
    isOpen = true,
    sourceId: WorkflowInvocationSourceId = 'generate'
  ) => {
    await act(() => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={workflowUiAdapter}>
              <WorkflowGraphPreviewProvider adapter={graphPreviewPort}>
                <GraphPreviewDialog
                  graphId="preview-graph-id"
                  isOpen={isOpen}
                  source={source}
                  sourceId={sourceId}
                  sourceLabel="Generate"
                  onOpenChange={onOpenChange}
                />
              </WorkflowGraphPreviewProvider>
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      );
    });
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn((_isOpen: boolean) => {});
    graphPreviewPort = createGraphPreviewPort();
    workflowUiAdapter = createWorkflowUiAdapter();
    downloads.downloadBlob.mockReset();
    downloads.downloadText.mockReset();
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

  it('renders summary rows, node count, and the live subtitle', async () => {
    await renderDialog(FIXTURE_SOURCE);

    const text = document.body.textContent ?? '';

    expect(text).toContain('This graph');
    expect(text).toContain('Gallery');
    expect(text).toContain('3');
    expect(text).toContain('Updates as you change settings.');
  });

  it('shows the seed notice banner', async () => {
    await renderDialog(FIXTURE_SOURCE);

    expect(document.body.textContent ?? '').toContain('Seed is randomized');
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
  });

  it('show node selects the seed node and inspector shows the randomized override', async () => {
    await renderDialog(FIXTURE_SOURCE);

    await clickButtonWithText('show node');

    const text = document.body.textContent ?? '';
    expect(text).toContain('integer');
    expect(text).toContain('seed');
    expect(text).toContain('regenerated each run');
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
});
