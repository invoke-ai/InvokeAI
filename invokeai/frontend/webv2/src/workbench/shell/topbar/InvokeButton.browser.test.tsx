import type { InvocationRoute } from '@workbench/invocationContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { InvocationIssue, InvocationState } from './useInvocationState';
import type * as UseTopbarShortcutModule from './useTopbarShortcut';

// Verify progress returns after hover ends unless focus-visible remains; mouse click-focus must not pin the play
// icon. Mock queue plumbing to isolate that gate.
const harness = vi.hoisted(() => ({
  bindings: {} as Record<string, { aria: string; display: string; parts: string[] }>,
  revealed: [] as string[],
  sources: [] as string[],
  progress: { activeItemIndex: 1, completedItemCount: 0, message: '', percentage: 0.42 as number | null },
  summary: { current: 1, remaining: 1, runningQueueItemId: 'item-1', total: 1 },
}));

vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({ progress: harness.progress, queueItems: [], summary: harness.summary }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: () => {} }) }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({ generation: { setSource: (sourceId: string) => harness.sources.push(sourceId) } }),
}));
vi.mock('./useTopbarShortcut', async (importOriginal) => ({
  ...(await importOriginal<typeof UseTopbarShortcutModule>()),
  useTopbarShortcutBinding: (commandId: string) => harness.bindings[commandId] ?? null,
}));
// The list's ways out are their own module's concern; here each layer issue offers one that records its layer.
vi.mock('./invokeIssueActions', () => ({
  useInvokeIssueActions: () => (issue: InvocationIssue) =>
    issue.target.kind === 'layer'
      ? [
          {
            id: 'show',
            label: 'Show layer',
            run: () => harness.revealed.push((issue.target as { layerId: string }).layerId),
          },
        ]
      : [],
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { count?: number; reason?: string; shortcut?: string }) =>
      options?.shortcut
        ? `${key}: ${options.shortcut}`
        : options?.count !== undefined
          ? `${key}(${[options.count, options.reason].filter((value) => value !== undefined).join(', ')})`
          : key,
  }),
}));

import { InvokeButton } from './InvokeButton';

const invocation: InvocationRoute = {
  destination: 'gallery',
  destinationLocked: false,
  sourceId: 'generate',
  sourceLocked: false,
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const makeInvocationState = (overrides: Partial<InvocationState> = {}): InvocationState => ({
  batchCount: 1,
  invocation,
  invoke: vi.fn(() => Promise.resolve()),
  issues: [],
  isPreparing: false,
  isValid: true,
  placedTypeIds: new Set<never>(),
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
  visibleTypeIds: new Set<never>(),
  ...overrides,
});

const renderInvokeButton = async (overrides: Partial<InvocationState> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const state = makeInvocationState(overrides);

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <InvokeButton state={state} />
      </ChakraProvider>
    )
  );

  return { button: host.querySelector('button') as HTMLButtonElement, state };
};

const LAYER_ISSUES: InvocationIssue[] = [
  { message: 'Control layer "Depth 1": Choose a model.', target: { kind: 'layer', layerId: 'control-2' } },
  { message: 'The server is disconnected.', target: { kind: 'connection' } },
];

const hasProgressRing = (button: HTMLButtonElement): boolean => button.querySelector('[role="progressbar"]') !== null;
const hasSpinner = (button: HTMLButtonElement): boolean => button.querySelector('.chakra-spinner') !== null;

beforeEach(() => {
  harness.bindings = {};
  harness.revealed = [];
  harness.sources = [];
  harness.progress.percentage = 0.42;
  harness.summary.total = 1;
  harness.summary.runningQueueItemId = 'item-1';
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('InvokeButton icon slot', () => {
  it('shows the progress ring while a batch runs and nothing is hovered or keyboard-focused', async () => {
    const { button } = await renderInvokeButton();

    expect(hasProgressRing(button)).toBe(true);
  });

  it('does not treat a mouse-click focus as keyboard focus: the ring returns once the pointer leaves', async () => {
    const { button } = await renderInvokeButton();

    await act(() => userEvent.click(button));
    // A plain click focuses the button (mousedown does that natively) but must
    // not arm `:focus-visible`, so it must not be the reason the ring is hidden
    // right after the click either.
    expect(document.activeElement).toBe(button);
    expect(button.matches(':focus-visible')).toBe(false);

    await act(() => userEvent.unhover(button));

    expect(hasProgressRing(button)).toBe(true);
  });

  it('suppresses the ring for real keyboard focus (:focus-visible) the same as hover', async () => {
    const { button } = await renderInvokeButton();

    await act(() => userEvent.tab());

    expect(document.activeElement).toBe(button);
    expect(button.matches(':focus-visible')).toBe(true);
    expect(hasProgressRing(button)).toBe(false);
  });

  it('uses the app spinner, not an indeterminate ring, while progress is unknown', async () => {
    harness.progress.percentage = null;
    const { button } = await renderInvokeButton();

    expect(hasSpinner(button)).toBe(true);
    expect(hasProgressRing(button)).toBe(false);
  });

  it('acknowledges canvas preparation immediately and ignores a second click', async () => {
    harness.summary.total = 0;
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({
      invoke,
      isPreparing: true,
      isValid: false,
      issues: [
        { message: 'Control layer "Depth 1": Choose a model.', target: { kind: 'layer', layerId: 'control-2' } },
      ],
    });

    expect(button.getAttribute('aria-disabled')).toBe('true');
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await expect.element(button).toHaveAccessibleDescription('topbar.invoke.preparing');
    expect(hasSpinner(button)).toBe(true);

    await act(() => button.click());

    expect(invoke).not.toHaveBeenCalled();
    expect(harness.revealed).toEqual([]);
  });
});

describe('InvokeButton front of queue', () => {
  const frontBinding = { aria: 'Control+Shift+Enter', display: 'Ctrl+Shift+Enter', parts: ['ctrl', 'shift', 'enter'] };

  it('queues at the front on Shift+click and normally on a plain click', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke });

    await act(() => userEvent.click(button));
    await userEvent.keyboard('{Shift>}');
    await act(() => userEvent.click(button));
    await userEvent.keyboard('{/Shift}');

    expect(invoke).toHaveBeenNthCalledWith(1, undefined);
    expect(invoke).toHaveBeenNthCalledWith(2, { prepend: true });
  });

  it('never queues while blocked, however it is clicked', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke, isValid: false, issues: LAYER_ISSUES });

    await act(() => userEvent.click(button));
    await userEvent.keyboard('{Shift>}');
    await act(() => userEvent.click(button));
    await userEvent.keyboard('{/Shift}');

    expect(invoke).not.toHaveBeenCalled();
  });

  it('names the live front-of-queue binding and Shift+click, and lists both bindings as shortcuts', async () => {
    harness.bindings = {
      'app.invoke': { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] },
      'app.invokeFront': frontBinding,
    };
    // Wide enough for the keycap hint.
    await page.viewport(1440, 900);
    const { button } = await renderInvokeButton();

    expect(button.getAttribute('aria-keyshortcuts')).toBe('Control+Enter Control+Shift+Enter');
    // The keycap hint shows at this width but stays out of the name; aria-keyshortcuts carries the binding.
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await act(() => userEvent.hover(button));

    await expect.element(page.getByText('topbar.invoke.invokeFrontHint: Ctrl+Shift+Enter')).toBeVisible();
  });

  it('names only Shift+click when the front-of-queue binding is cleared', async () => {
    const { button } = await renderInvokeButton();

    await act(() => userEvent.hover(button));

    await expect.element(page.getByText('topbar.invoke.invokeFrontHintNoShortcut')).toBeVisible();
  });
});

describe('InvokeButton issues', () => {
  it('counts the issues in place of the shortcut and steps back from the primary color', async () => {
    harness.bindings = { 'app.invoke': { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] } };
    const { button } = await renderInvokeButton({ isValid: false, issues: LAYER_ISSUES });

    await expect.element(button).toHaveAccessibleName('topbar.invoke.issueCount(2)');
    expect(button.querySelector('kbd')).toBeNull();
    // Blocked is not disabled: activating it opens the list.
    expect(button.getAttribute('aria-disabled')).not.toBe('true');
    expect(button.getAttribute('aria-haspopup')).toBe('dialog');
    expect(button.dataset.blocked).toBe('');
  });

  it('previews the issues on hover without taking focus', async () => {
    const { button } = await renderInvokeButton({ isValid: false, issues: LAYER_ISSUES });

    await act(() => userEvent.hover(button));

    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).toBeVisible();
    await expect.element(page.getByText('The server is disconnected.')).toBeVisible();
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).not.toBe(page.getByRole('button', { name: 'Show layer' }).element());
  });

  it('opens the list on click or Enter, with each issue’s way out, and never invokes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke, isValid: false, issues: LAYER_ISSUES });

    button.focus();
    await act(() => userEvent.keyboard('{Enter}'));

    const show = page.getByRole('button', { name: 'Show layer' });
    await expect.element(show).toHaveFocus();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    // The lost connection is listed with nothing to press.
    await expect.element(page.getByText('The server is disconnected.')).toBeVisible();

    await act(() => show.click());

    expect(harness.revealed).toEqual(['control-2']);
    expect(invoke).not.toHaveBeenCalled();
    // A jump closes the list and leaves focus where the jump put it.
    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).not.toBeInTheDocument();
  });

  const WORKFLOW_ROUTE = {
    invocation: { ...invocation, sourceId: 'workflow' as const },
    isValid: false,
    placedTypeIds: new Set(['generate', 'workflow'] as const),
    sources: [
      { label: 'Generate', sourceId: 'generate' as const, typeId: 'generate' as const },
      { label: 'Workflow', sourceId: 'workflow' as const, typeId: 'workflow' as const },
    ],
  };

  // With no way out to press, a click lands on the source the route runs, never on one the user did not choose.
  it('pins focus on the selected source when no issue offers an action', async () => {
    const { button } = await renderInvokeButton({
      ...WORKFLOW_ROUTE,
      issues: [{ message: 'The server is disconnected.', target: { kind: 'connection' } }],
    });

    await act(() => userEvent.click(button));

    await expect.poll(() => (document.activeElement as HTMLInputElement | null)?.value).toBe('workflow');
    expect((document.activeElement as HTMLInputElement).checked).toBe(true);
  });

  it('gives focus back to Invoke on Escape, and on Tab past its last control', async () => {
    const { button } = await renderInvokeButton({ isValid: false, issues: LAYER_ISSUES });

    await act(() => userEvent.click(button));
    await expect.element(page.getByRole('button', { name: 'Show layer' })).toHaveFocus();
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.element(button).toHaveFocus();
    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).not.toBeInTheDocument();

    await act(() => userEvent.click(button));
    await expect.element(page.getByRole('button', { name: 'Show layer' })).toHaveFocus();
    await act(() => userEvent.tab());

    await expect.element(button).toHaveFocus();
    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).not.toBeInTheDocument();
  });

  it('closes a hover preview once the pointer leaves', async () => {
    const { button } = await renderInvokeButton({ isValid: false, issues: LAYER_ISSUES });

    await act(() => userEvent.hover(button));
    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).toBeVisible();
    await act(() => userEvent.unhover(button));

    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).not.toBeInTheDocument();
  });

  it('switches the source among those the layout holds, from the list itself', async () => {
    const { button } = await renderInvokeButton({
      invocation: { ...invocation, sourceId: 'workflow' },
      isValid: false,
      issues: [
        {
          message: 'The Workflow widget is not in this layout. Add it, or run another source.',
          target: { kind: 'mountSource', sourceId: 'workflow' },
        },
      ],
      placedTypeIds: new Set(['generate', 'canvas']),
      sources: [
        { label: 'Generate', sourceId: 'generate', typeId: 'generate' },
        { label: 'Canvas', sourceId: 'canvas', typeId: 'canvas' },
        { label: 'Upscale', sourceId: 'upscale', typeId: 'upscale' },
        { label: 'Workflow', sourceId: 'workflow', typeId: 'workflow' },
      ],
    });

    await act(() => userEvent.click(button));
    const sourceGroup = page.getByRole('radiogroup', { name: 'topbar.routing.source' });

    await expect.element(sourceGroup).toBeVisible();
    // Upscale is in no region or window, so it is not offered; the missing Workflow stays, as the current choice.
    expect(sourceGroup.getByText('widgets.labels.upscale').query()).toBeNull();
    await act(() => sourceGroup.getByText('widgets.labels.generate').click());

    expect(harness.sources).toEqual(['generate']);
  });
});
