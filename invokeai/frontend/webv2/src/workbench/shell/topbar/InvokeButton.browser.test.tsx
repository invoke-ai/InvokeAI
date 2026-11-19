import type * as UseShortcutBindingModule from '@workbench/hotkeys/useShortcutBinding';
import type { InvocationRoute } from '@workbench/invocationContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { InvocationIssue, InvocationState } from './useInvocationState';

const harness = vi.hoisted(() => ({
  bindings: {} as Record<string, { aria: string; display: string; parts: string[] }>,
  destinations: [] as string[],
  lockToggles: 0,
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
  useWorkbenchCommands: () => ({
    generation: {
      setDestination: (destination: string) => harness.destinations.push(destination),
      setSource: (sourceId: string) => harness.sources.push(sourceId),
      toggleRoutingLock: () => harness.lockToggles++,
    },
  }),
}));
vi.mock('@workbench/hotkeys/useShortcutBinding', async (importOriginal) => ({
  ...(await importOriginal<typeof UseShortcutBindingModule>()),
  useShortcutBinding: (commandId: string) => harness.bindings[commandId] ?? null,
}));
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
    t: (
      key: string,
      options?: { count?: number; destination?: string; reason?: string; shortcut?: string; source?: string }
    ) =>
      options?.shortcut
        ? `${key}: ${options.shortcut}`
        : options?.source !== undefined
          ? `${key}(${options.source}, ${options.destination})`
          : options?.count !== undefined
            ? `${key}(${[options.count, options.reason].filter((value) => value !== undefined).join(', ')})`
            : key,
  }),
}));

import { InvokeButton, PREVIEW_CLOSE_DELAY_MS } from './InvokeButton';

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

const renderState = (state: InvocationState) =>
  act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <InvokeButton state={state} />
      </ChakraProvider>
    )
  );

const renderInvokeButton = async (overrides: Partial<InvocationState> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const state = makeInvocationState(overrides);

  await renderState(state);

  return {
    button: host.querySelector('button') as HTMLButtonElement,
    rerender: (next: Partial<InvocationState>) => renderState(makeInvocationState({ ...overrides, ...next })),
    state,
  };
};

const LAYER_ISSUES: InvocationIssue[] = [
  { message: 'Control layer "Depth 1": Choose a model.', target: { kind: 'layer', layerId: 'control-2' } },
  { message: 'The server is disconnected.', target: { kind: 'connection' } },
];

const hasProgressRing = (button: HTMLButtonElement): boolean => button.querySelector('[role="progressbar"]') !== null;
const hasSpinner = (button: HTMLButtonElement): boolean => button.querySelector('.chakra-spinner') !== null;

beforeEach(() => {
  harness.bindings = {};
  harness.destinations = [];
  harness.lockToggles = 0;
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
    await expect
      .element(button)
      .toHaveAccessibleDescription(
        'topbar.invoke.preparing topbar.invoke.routeDescription(widgets.labels.generate, widgets.labels.gallery) topbar.invoke.routeHint'
      );
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

  describe('while Shift is held', () => {
    const frontIcon = (button: HTMLButtonElement) => button.querySelector('[data-invoke-front-icon]');

    beforeEach(async () => {
      harness.bindings = {
        'app.invoke': { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] },
        'app.invokeFront': frontBinding,
      };
      await page.viewport(1440, 900);
    });

    afterEach(() => userEvent.keyboard('{/Shift}'));

    it('shows the front-of-queue icon over a ready button, at the same width, until released', async () => {
      const { button } = await renderInvokeButton();

      await act(() => userEvent.hover(button));
      const width = button.getBoundingClientRect().width;
      expect(frontIcon(button)).toBeNull();

      await act(() => userEvent.keyboard('{Shift>}'));

      expect(frontIcon(button)).not.toBeNull();
      expect(button.getBoundingClientRect().width).toBe(width);
      await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');

      await act(() => userEvent.keyboard('{/Shift}'));

      expect(frontIcon(button)).toBeNull();
    });
  });
});

describe('InvokeButton issues', () => {
  it('counts the issues in place of the shortcut and steps back from the primary color', async () => {
    harness.bindings = { 'app.invoke': { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] } };
    const { button } = await renderInvokeButton({ isValid: false, issues: LAYER_ISSUES });

    await expect.element(button).toHaveAccessibleName('topbar.invoke.issueCount(2)');
    expect(button.querySelector('kbd')).toBeNull();
    expect(button.getAttribute('aria-disabled')).not.toBe('true');
    expect(button.getAttribute('aria-haspopup')).toBe('dialog');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(button.dataset.blocked).toBe('');
  });

  it('opens the list on click or Enter, with each issue’s way out, and never invokes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke, isValid: false, issues: LAYER_ISSUES });

    button.focus();
    await act(() => userEvent.keyboard('{Enter}'));

    const show = page.getByRole('button', { name: 'Show layer' });
    await expect.element(show).toHaveFocus();
    expect(button.getAttribute('aria-expanded')).toBe('true');
    await expect.element(page.getByText('The server is disconnected.')).toBeVisible();

    await act(() => show.click());

    expect(harness.revealed).toEqual(['control-2']);
    expect(invoke).not.toHaveBeenCalled();
    await expect.element(page.getByText('Control layer "Depth 1": Choose a model.')).not.toBeInTheDocument();
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
    await act(
      () =>
        new Promise((resolve) => {
          setTimeout(resolve, 400);
        })
    );
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
    expect(sourceGroup.getByText('widgets.labels.upscale').query()).toBeNull();
    await act(() => sourceGroup.getByText('widgets.labels.generate').click());

    expect(harness.sources).toEqual(['generate']);
  });
});

describe('InvokeButton route card', () => {
  const MULTI_SOURCE = {
    placedTypeIds: new Set(['generate', 'canvas'] as const),
    sources: [
      { label: 'Generate', sourceId: 'generate' as const, typeId: 'generate' as const },
      { label: 'Canvas', sourceId: 'canvas' as const, typeId: 'canvas' as const },
    ],
  };
  const sourceGroup = () => page.getByRole('radiogroup', { name: 'topbar.routing.source' });
  const destinationGroup = () => page.getByRole('radiogroup', { name: 'topbar.routing.destination' });
  const cardMode = () => document.querySelector('[data-invoke-card-mode]')?.getAttribute('data-invoke-card-mode');
  const waitPastCloseDelay = () =>
    act(
      () =>
        new Promise<void>((resolve) => {
          setTimeout(resolve, PREVIEW_CLOSE_DELAY_MS + 150);
        })
    );

  it('opens pinned on ArrowDown while valid, focusing the selected source, and never invokes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ ...MULTI_SOURCE, invoke });

    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));

    await expect.poll(() => (document.activeElement as HTMLInputElement | null)?.value).toBe('generate');
    expect(sourceGroup().element().contains(document.activeElement)).toBe(true);
    expect(button.hasAttribute('aria-expanded')).toBe(false);
    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);
    await expect.element(page.getByRole('dialog')).toHaveAccessibleName('topbar.invoke.addToQueue');
    expect(invoke).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{Escape}'));
    await expect.element(button).toHaveFocus();
    await expect.element(sourceGroup()).not.toBeInTheDocument();
  });

  it('opens pinned on right-click instead of the native menu, and never invokes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke });
    const contextMenu = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });

    await act(() => userEvent.hover(button));
    await act(() => button.dispatchEvent(contextMenu));

    expect(contextMenu.defaultPrevented).toBe(true);
    await expect.element(destinationGroup()).toBeVisible();
    expect(cardMode()).toBe('pinned');
    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);

    await act(() => userEvent.unhover(button));
    await act(() => userEvent.hover(document.body));
    await waitPastCloseDelay();
    await expect.element(destinationGroup()).toBeVisible();
    expect(cardMode()).toBe('pinned');
    expect(invoke).not.toHaveBeenCalled();
  });

  it('locks the route from the card and marks a locked route on Invoke itself', async () => {
    const { button, rerender } = await renderInvokeButton();

    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    const lock = page.getByRole('button', { name: 'topbar.routing.lockRouting' });
    await expect.element(lock).toHaveAttribute('aria-pressed', 'false');
    expect(button.querySelector('[data-routing-lock-indicator]')).toBeNull();

    await act(() => lock.click());
    expect(harness.lockToggles).toBe(1);
    await rerender({ invocation: { ...invocation, destinationLocked: true, sourceLocked: true } });

    await expect.element(lock).toHaveAttribute('aria-pressed', 'true');
    await expect.element(lock).toHaveFocus();
    expect(button.querySelector('[data-routing-lock-indicator]')).not.toBeNull();
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await expect
      .element(button)
      .toHaveAccessibleDescription(
        'topbar.invoke.routeDescriptionLocked(widgets.labels.generate, widgets.labels.gallery) topbar.invoke.routeHint'
      );
  });
});
