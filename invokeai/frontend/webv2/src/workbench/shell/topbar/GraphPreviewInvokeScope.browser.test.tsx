import type { GraphPreviewSourceState, InvocationTemplatesSnapshot } from '@features/workflow/contracts';
import type { WorkflowGraphPreviewPort, WorkflowUiAdapter } from '@features/workflow/react';
import type * as UseShortcutBindingModule from '@workbench/hotkeys/useShortcutBinding';
import type { InvocationIssue, InvocationState } from '@workbench/shell/topbar/useInvocationState';
import type * as UseInvocationStateModule from '@workbench/shell/topbar/useInvocationState';

import { ChakraProvider } from '@chakra-ui/react';
import { GraphPreviewDialog } from '@features/workflow/preview';
import { WorkflowGraphPreviewProvider, WorkflowUiProvider } from '@features/workflow/react';
import { createProjectGraph } from '@features/workflow/utility';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { GraphPreviewInvokeScope } from './GraphPreviewInvokeScope';
import { InvokeButton } from './InvokeButton';

/**
 * The View graph dialog's Invoke: the top bar's control bound to the previewed source's route. The route state is the
 * workbench's (`useInvocationState`, tested on its own); here it is the state a blocked or ready Canvas route
 * publishes, so the control, its card and the dialog meet as they do in the app.
 */
const harness = vi.hoisted(() => ({
  requestedSources: [] as (string | undefined)[],
  revealed: [] as string[],
  fixed: [] as string[],
  state: null as unknown as InvocationState,
}));

vi.mock('@workbench/shell/topbar/useInvocationState', async (importOriginal) => ({
  ...(await importOriginal<typeof UseInvocationStateModule>()),
  useInvocationState: ({ sourceId }: { sourceId?: string } = {}) => {
    harness.requestedSources.push(sourceId);
    return harness.state;
  },
}));
vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({ progress: null, queueItems: [], summary: { current: 0, remaining: 0, total: 0 } }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: () => {} }) }));
vi.mock('@workbench/hotkeys/useShortcutBinding', async (importOriginal) => ({
  ...(await importOriginal<typeof UseShortcutBindingModule>()),
  // The app's Invoke shortcut is bound, as it is by default.
  useShortcutBinding: (commandId: string) =>
    commandId === 'app.invoke' ? { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] } : null,
}));
// Each layer issue offers a fix that stays and a way to the layer that leads away, as the real actions do.
vi.mock('@workbench/shell/topbar/invokeIssueActions', () => ({
  useInvokeIssueActions: () => (issue: InvocationIssue) =>
    issue.target.kind === 'layer'
      ? [
          { id: 'fix', label: 'Fix', run: () => harness.fixed.push('control-2'), stays: true },
          { id: 'show', label: 'Show layer', run: () => harness.revealed.push('control-2') },
        ]
      : [],
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: vi.fn(() => Promise.resolve()),
  ensureStartersLoaded: vi.fn(),
  useActiveInstallSources: () => new Set<string>(),
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector({ models: [] }),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) => selector({ response: { starter_models: [] } }),
}));

const TEMPLATES_SNAPSHOT = vi.hoisted((): InvocationTemplatesSnapshot => ({
  error: null,
  status: 'loaded',
  templates: {},
}));

vi.mock('@features/workflow/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationTemplatesSnapshot: () => TEMPLATES_SNAPSHOT,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number }) => (options?.count === undefined ? key : `${key}(${options.count})`),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LAYER_ISSUE: InvocationIssue = {
  message: 'Control layer "Depth 1": Choose a model.',
  target: { kind: 'layer', layerId: 'control-2' },
};

const makeState = (overrides: Partial<InvocationState> = {}): InvocationState => ({
  batchCount: 1,
  invocation: { destination: 'gallery', destinationLocked: false, sourceId: 'canvas', sourceLocked: true },
  invoke: vi.fn(() => Promise.resolve()),
  issues: [],
  isPreparing: false,
  isValid: true,
  placedTypeIds: new Set(),
  promptExpansion: {
    count: 1,
    error: null,
    isDynamic: false,
    isError: false,
    isLoading: false,
    isRetrying: false,
    prompts: [],
    retry: vi.fn(),
  },
  sourceValues: {},
  sources: [],
  visibleTypeIds: new Set(),
  ...overrides,
});

const SOURCE: GraphPreviewSourceState = {
  destinationLabel: 'Gallery',
  graph: {
    edges: [],
    id: 'canvas-graph',
    nodes: [{ id: 'decode', inputs: {}, type: 'l2i' }],
    version: 1,
  },
  invalidReasons: [],
  isLive: false,
  notices: [],
  summaryRows: [],
};

const project = createProjectGraph('workflow-1');
const ADAPTER = {
  capabilities: { getSnapshot: () => ({ canUseCache: true }), subscribe: () => () => {} },
  commands: { addWorkflow: vi.fn() },
  getProjectGraph: () => project,
  nodeExecution: { get: () => null, getOrigin: () => null, subscribe: () => () => {}, subscribeOrigin: () => () => {} },
  notifications: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
  performance: { mark: vi.fn(), measure: vi.fn(), time: (_n: string, _s: unknown, run: () => unknown) => run() },
  preferences: {
    getSnapshot: () => ({ reduceMotion: false, themeId: 'classic', workflowEdgeStyle: 'curved' }),
    subscribe: () => () => {},
  },
  project: {
    getSnapshot: () => ({
      activeWorkflow: { document: project },
      id: 'project-1',
      projectGraph: project,
      workflows: [],
    }),
    subscribe: () => () => {},
  },
  widgets: { open: vi.fn(), patchValues: vi.fn() },
} as unknown as WorkflowUiAdapter;
const PORT: WorkflowGraphPreviewPort = {
  focusSource: vi.fn(),
  openDocumentInNewProject: vi.fn(),
  openWorkflowEditor: vi.fn(),
};

const Preview = ({ onOpenChange }: { onOpenChange: (isOpen: boolean) => void }) => (
  <GraphPreviewInvokeScope
    // oxlint-disable-next-line react-perf/jsx-no-new-function-as-prop -- one preview per test render.
    render={(invoke) => (
      <GraphPreviewDialog
        graphId="canvas-graph"
        invoke={invoke}
        isOpen
        source={SOURCE}
        sourceId="canvas"
        sourceLabel="Canvas"
        onOpenChange={onOpenChange}
      />
    )}
    sourceId="canvas"
    onOpenChange={onOpenChange}
  />
);

describe('a graph preview’s Invoke', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onOpenChange: ReturnType<typeof vi.fn<(isOpen: boolean) => void>>;
  /** The top bar's own Invoke, for the active route, mounted beside the dialog as in the app. */
  let topBarState: InvocationState;

  const render = (state: InvocationState) => {
    harness.state = state;

    return act(() =>
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <InvokeButton state={topBarState} />
            <WorkflowUiProvider adapter={ADAPTER}>
              <WorkflowGraphPreviewProvider adapter={PORT}>
                <Preview onOpenChange={onOpenChange} />
              </WorkflowGraphPreviewProvider>
            </WorkflowUiProvider>
          </ChakraProvider>
        </StrictMode>
      )
    );
  };

  const footer = () => document.querySelector<HTMLElement>('[data-graph-preview-footer]');
  const dialogContent = () => document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]');
  const previewInvoke = () => footer()?.querySelector<HTMLButtonElement>('button[aria-keyshortcuts]') ?? null;
  const card = () => document.querySelector<HTMLElement>('[data-invoke-card-mode]');
  const settle = () =>
    act(async () => {
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 50);
      });
    });

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    onOpenChange = vi.fn<(isOpen: boolean) => void>();
    topBarState = makeState({
      invocation: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
    });
    harness.requestedSources = [];
    harness.revealed = [];
    harness.fixed = [];
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('is the top bar’s control for the previewed source, the same height as the footer’s other button', async () => {
    const previous = { height: window.innerHeight, width: window.innerWidth };
    await page.viewport(1288, 942);
    onTestFinished(() => page.viewport(previous.width, previous.height));
    await render(makeState());

    expect(harness.requestedSources).toContain('canvas');
    expect(previewInvoke()?.textContent).toContain('topbar.invoke.invoke');
    const editAsWorkflow = footer()!.querySelector<HTMLElement>('[data-graph-preview-edit-as-workflow]')!;
    expect(previewInvoke()!.getBoundingClientRect().height).toBe(editAsWorkflow.getBoundingClientRect().height);
    // On one line with the status.
    const status = footer()!.querySelector<HTMLElement>('[data-graph-preview-status]')!.getBoundingClientRect();
    const button = previewInvoke()!.getBoundingClientRect();
    expect(Math.abs(status.top + status.height / 2 - (button.top + button.height / 2))).toBeLessThan(2);
  });

  it('advertises no app shortcut, which does nothing in a dialog, while the top bar’s Invoke still does', async () => {
    await render(makeState());

    const topBar = host.querySelector<HTMLButtonElement>(':scope > button')!;

    expect(topBar.getAttribute('aria-keyshortcuts')).toBe('Control+Enter ArrowDown');
    expect(topBar.querySelector('kbd')).not.toBeNull();
    expect(previewInvoke()?.getAttribute('aria-keyshortcuts')).toBe('ArrowDown');
    expect(previewInvoke()?.querySelector('kbd')).toBeNull();
  });

  it('runs the previewed route once and closes, leaving the top bar’s Invoke alone', async () => {
    const state = makeState({ invoke: vi.fn((options) => Promise.resolve(options?.onStarted?.())) });
    await render(state);

    await act(() => userEvent.click(previewInvoke()!));

    expect(state.invoke).toHaveBeenCalledTimes(1);
    expect(topBarState.invoke).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('opens its issue card above the dialog, with the route fixed, and closes the dialog for a way out', async () => {
    await render(makeState({ issues: [LAYER_ISSUE], isValid: false }));

    expect(previewInvoke()?.textContent).toContain('topbar.invoke.issueCount(1)');
    await act(() => userEvent.click(previewInvoke()!));
    await vi.waitFor(() => expect(card()?.textContent).toContain('Choose a model.'));
    await settle();

    // Above the dialog: what is under the card's centre is the card.
    const box = card()!.getBoundingClientRect();
    expect(card()!.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2))).toBe(true);
    expect(dialogContent()?.contains(card())).toBe(false);
    // The route is the preview's: shown, with nothing to switch or lock.
    expect(card()!.querySelector('[data-invoke-route-fixed]')).not.toBeNull();
    expect(card()!.querySelector('[data-routing-lock]')).toBeNull();
    // Focus went into the card, past the dialog's trap.
    expect(card()!.contains(document.activeElement)).toBe(true);

    // A fix acts in place: the card and the dialog stay.
    await act(() => userEvent.click([...card()!.querySelectorAll('button')].find((b) => b.textContent === 'Fix')!));
    expect(harness.fixed).toEqual(['control-2']);
    expect(onOpenChange).not.toHaveBeenCalled();

    await act(() =>
      userEvent.click([...card()!.querySelectorAll('button')].find((b) => b.textContent === 'Show layer')!)
    );

    expect(harness.revealed).toEqual(['control-2']);
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('closes on Escape its card first, then the dialog', async () => {
    await render(makeState({ issues: [LAYER_ISSUE], isValid: false }));
    await act(() => userEvent.click(previewInvoke()!));
    await vi.waitFor(() => expect(card()).not.toBeNull());
    await settle();

    await act(() => userEvent.keyboard('{Escape}'));
    await act(async () => {
      await vi.waitFor(() => expect(card()).toBeNull());
    });

    expect(onOpenChange).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{Escape}'));

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
