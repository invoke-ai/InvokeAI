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
  'graphPreview.compiledFrom': 'Compiled from {{source}}.',
  'graphPreview.copied': 'Copied',
  'graphPreview.copyJson': 'Copy JSON',
  'graphPreview.destination': 'Destination',
  'graphPreview.graph': 'Graph',
  'graphPreview.graphJsonLabel': '{{title}} graph JSON',
  'graphPreview.invalidTitle': "This graph can't compile yet.",
  'graphPreview.invokeRoute': 'Invoke {{route}}',
  'graphPreview.list': 'List',
  'graphPreview.liveHint': 'Updates as you change settings.',
  'graphPreview.noCompiledGraph': 'No compiled graph is available for "{{graphId}}" yet.',
  'graphPreview.nodes': 'Nodes',
  'graphPreview.selectNode': 'Select a node for details.',
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
  graph: FIXTURE_GRAPH,
  invalidReasons: [],
  isLive: true,
  notices: [
    { id: 'seed-random', message: 'Seed is randomized. This graph runs differently each time.', nodeId: 'seed' },
  ],
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

  const renderDialog = async (source: GraphPreviewSourceState) => {
    const workflowUiAdapter = createWorkflowUiAdapter();
    const graphPreviewPort = createGraphPreviewPort();

    await act(() => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WorkflowUiProvider adapter={workflowUiAdapter}>
              <WorkflowGraphPreviewProvider adapter={graphPreviewPort}>
                <GraphPreviewDialog
                  graphId="preview-graph-id"
                  isOpen
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
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

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
});
