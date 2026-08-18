import type { GraphPreviewSourceState, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
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
  'graphPreview.edges': 'Edges',
  'graphPreview.edgesIn': 'in · {{count}} inputs from {{sources}}',
  'graphPreview.edgesInNone': 'in · none',
  'graphPreview.edgesOut': 'out · {{field}} → {{target}}',
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
  'graphPreview.resolvedInputs': 'Resolved inputs',
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

const createWorkflowUiAdapter = (): WorkflowUiAdapter =>
  ({
    capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
    commands: {},
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
      getSnapshot: () => ({
        galleryValues: {},
        graphHistory: [],
        id: 'project-1',
        isWorkflowRunning: false,
        projectGraph: { edges: [], nodes: [], version: 1 as const },
        workflowValues: {},
      }),
      subscribe: () => () => {},
    },
    registerModalHotkeyLayer: vi.fn(() => vi.fn()),
    widgets: { open: vi.fn(), patchValues: vi.fn() },
  }) as unknown as WorkflowUiAdapter;

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

  const renderDialog = async (source: GraphPreviewSourceState, isOpen = true) => {
    const workflowUiAdapter = createWorkflowUiAdapter();

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
                  sourceId="generate"
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
  });

  afterEach(async () => {
    await act(() => root.unmount());
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
});
