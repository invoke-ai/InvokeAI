import type { InvocationTemplate, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowUiAdapter } from '@features/workflow/ui/WorkflowUiContext';
import type * as DeferredModuleModule from '@platform/react/deferredModule';

/* eslint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop -- each test mounts a fresh flow */
import { ChakraProvider } from '@chakra-ui/react';
import { WorkflowUiProvider } from '@features/workflow/ui/WorkflowUiContext';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { ReactFlow } from '@xyflow/react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { toFlowNodes } from './flowAdapters';
import { InvocationFlowNode } from './InvocationFlowNode';

import '@xyflow/react/dist/style.css';

// The later open's real import resolves from the module cache rather than an on-demand transform under suite load.
await import('./NodePanels');

// The panel body's chunk fails to load for the first open, as a stale or dropped chunk does: its import rejects. The
// rejection comes from the deferred module's import seam, because a `vi.mock` factory that rejects is not an import
// failure in browser mode: Vitest's module interception reports it as an unhandled error and never answers the request.
const panelLoad = vi.hoisted(() => ({ fails: false, rejected: 0 }));
vi.mock('@platform/react/deferredModule', async (importOriginal) => {
  const actual = await importOriginal<typeof DeferredModuleModule>();

  return {
    ...actual,
    // The node's panels are the only deferred module this view renders.
    createDeferredModule: <Module,>(importModule: () => Promise<Module>) =>
      actual.createDeferredModule(() => {
        if (!panelLoad.fails) {
          return importModule();
        }
        panelLoad.rejected += 1;
        return Promise.reject(new TypeError('Failed to fetch dynamically imported module'));
      }),
  };
});
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) =>
      key === 'nodes.status.label' ? `Status: ${String(options?.status)}` : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const template: InvocationTemplate = {
  category: 'test',
  classification: 'stable',
  description: '',
  inputs: {},
  nodePack: 'invokeai',
  outputType: 'test',
  outputs: {},
  tags: [],
  title: 'Preview',
  type: 'preview',
  useCache: true,
  version: '1.1.0',
};
// An older version than its template, so the node has a status to report.
const node: WorkflowInvocationNode = {
  data: {
    inputs: {},
    isIntermediate: true,
    isOpen: true,
    label: '',
    nodePack: 'invokeai',
    notes: '',
    type: 'preview',
    useCache: true,
    version: '1.0.0',
  },
  id: 'node',
  position: { x: 20, y: 60 },
  type: 'invocation',
};

const adapter = {
  capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
  commands: { editGraph: vi.fn() },
  nodeExecution: {
    get: () => null,
    getOrigin: () => null,
    subscribe: () => () => {},
    subscribeOrigin: () => () => {},
  },
  project: { getSnapshot: () => ({ id: 'p' }), subscribe: () => () => {} },
} as unknown as WorkflowUiAdapter;

describe('node header panels when their body fails', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width: 480px; height: 360px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('shows the button’s label instead of crashing the editor, and the full panel on a later open', async () => {
    panelLoad.fails = true;
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <WorkflowUiProvider adapter={adapter}>
            <ReactFlow
              nodes={toFlowNodes({ ...createProjectGraph('panels'), nodes: [node] }, [], { preview: template })}
              nodeTypes={{ invocation: InvocationFlowNode }}
            />
          </WorkflowUiProvider>
        </ChakraProvider>
      )
    );
    const statusButton = () => host.querySelector<HTMLElement>('[data-node-status-indicator="true"]')!;
    const panel = () => document.querySelector('[data-scope="tooltip"][data-part="content"]');
    const label = statusButton().getAttribute('aria-label')!;

    await act(() => userEvent.hover(statusButton()));
    // The label shows while loading too; this is it standing in for a load that failed.
    await vi.waitFor(() => expect(panelLoad.rejected).toBeGreaterThan(0));
    await vi.waitFor(() => expect(panel()?.textContent).toBe(label));
    // The node is still there and still answers.
    expect(host.querySelector('.react-flow__node[data-id="node"]')).not.toBeNull();
    await act(() => userEvent.unhover(statusButton()));
    await vi.waitFor(() => expect(panel()).toBeNull());
    panelLoad.fails = false;

    await act(() => userEvent.hover(statusButton()));
    // The panel body: the version line on its own, not the button's "Status: …" label.
    await vi.waitFor(() => expect(panel()?.textContent).toBe('nodes.nodeUpdateAvailable'));
  });
});
