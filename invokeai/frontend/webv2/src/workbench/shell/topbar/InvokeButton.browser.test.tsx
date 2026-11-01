import type * as UseShortcutBindingModule from '@workbench/hotkeys/useShortcutBinding';
import type { InvocationRoute } from '@workbench/invocationContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { InvocationIssue, InvocationState } from './useInvocationState';

// Verify progress returns after hover ends unless focus-visible remains; mouse click-focus must not pin the play
// icon. Mock queue plumbing to isolate that gate.
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
    /** Re-renders with the state the store would publish next. */
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
    // The reason leads, and the route still follows it.
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

  it('names the live front-of-queue binding and Shift+click, and lists both bindings as shortcuts', async () => {
    harness.bindings = {
      'app.invoke': { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] },
      'app.invokeFront': frontBinding,
    };
    // Wide enough for the keycap hint.
    await page.viewport(1440, 900);
    const { button } = await renderInvokeButton();

    expect(button.getAttribute('aria-keyshortcuts')).toBe('Control+Enter Control+Shift+Enter ArrowDown');
    // The keycap hint shows at this width but stays out of the name; aria-keyshortcuts carries the binding.
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await act(() => userEvent.hover(button));

    await expect.element(page.getByText('topbar.invoke.invokeFrontHint: Ctrl+Shift+Enter')).toBeVisible();
  });

  describe('while Shift is held', () => {
    const frontIcon = (button: HTMLButtonElement) => button.querySelector('[data-invoke-front-icon]');

    beforeEach(async () => {
      harness.bindings = {
        'app.invoke': { aria: 'Control+Enter', display: 'Ctrl+Enter', parts: ['ctrl', 'enter'] },
        'app.invokeFront': frontBinding,
      };
      // Wide enough for the keycap hint.
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

    it('forgets a Shift released away from the button', async () => {
      const { button } = await renderInvokeButton();

      await act(() => userEvent.hover(button));
      await act(() => userEvent.keyboard('{Shift>}'));
      expect(frontIcon(button)).not.toBeNull();
      await act(() => userEvent.unhover(button));
      await act(() => userEvent.keyboard('{/Shift}'));
      await act(() => userEvent.hover(button));

      expect(frontIcon(button)).toBeNull();
    });

    it('leaves the icon alone when Shift is held away from the button', async () => {
      const { button } = await renderInvokeButton();

      await act(() => userEvent.keyboard('{Shift>}'));

      // Running and not hovered: the ring stays.
      expect(hasProgressRing(button)).toBe(true);
      expect(frontIcon(button)).toBeNull();
    });

    it('leaves a blocked button’s icon alone', async () => {
      const { button } = await renderInvokeButton({ isValid: false, issues: LAYER_ISSUES });

      await act(() => userEvent.hover(button));
      await act(() => userEvent.keyboard('{Shift>}'));

      expect(frontIcon(button)).toBeNull();
      expect(button.querySelector('svg.lucide-circle-alert')).not.toBeNull();
    });
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
    // Activation opens the list, so the button announces it.
    expect(button.getAttribute('aria-haspopup')).toBe('dialog');
    expect(button.getAttribute('aria-expanded')).toBe('false');
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
    // Returning keyboard focus must not schedule the hover/focus preview and reopen the dismissed card.
    await act(
      () =>
        new Promise((resolve) => {
          setTimeout(resolve, 400);
        })
    );
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

  it('announces no popup while activation queues, but names the route and the way into it', async () => {
    const { button } = await renderInvokeButton();

    expect(button.hasAttribute('aria-haspopup')).toBe(false);
    expect(button.hasAttribute('aria-expanded')).toBe(false);
    expect(button.getAttribute('aria-controls')).toBeNull();
    expect(button.getAttribute('aria-keyshortcuts')?.split(' ')).toContain('ArrowDown');
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await expect
      .element(button)
      .toHaveAccessibleDescription(
        'topbar.invoke.routeDescription(widgets.labels.generate, widgets.labels.gallery) topbar.invoke.routeHint'
      );
  });

  it('describes a Video route as going to the Gallery, whatever destination it holds', async () => {
    const { button } = await renderInvokeButton({
      invocation: { ...invocation, destination: 'canvas', sourceId: 'video' },
    });

    await expect
      .element(button)
      .toHaveAccessibleDescription(
        'topbar.invoke.routeDescription(widgets.labels.video, widgets.labels.gallery) topbar.invoke.routeHint'
      );
  });

  it('opens pinned on ArrowDown while valid, focusing the selected source, and never invokes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ ...MULTI_SOURCE, invoke });

    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));

    await expect.poll(() => (document.activeElement as HTMLInputElement | null)?.value).toBe('generate');
    expect(sourceGroup().element().contains(document.activeElement)).toBe(true);
    expect(button.hasAttribute('aria-expanded')).toBe(false);
    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);
    // The card the description's Down Arrow hint points to is named by its title.
    await expect.element(page.getByRole('dialog')).toHaveAccessibleName('topbar.invoke.addToQueue');
    expect(invoke).not.toHaveBeenCalled();

    await act(() => userEvent.keyboard('{Escape}'));
    await expect.element(button).toHaveFocus();
    await expect.element(sourceGroup()).not.toBeInTheDocument();
  });

  it('opens pinned on ArrowDown with the first issue’s way out focused when blocked', async () => {
    const { button } = await renderInvokeButton({ ...MULTI_SOURCE, isValid: false, issues: LAYER_ISSUES });

    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));

    await expect.element(page.getByRole('button', { name: 'Show layer' })).toHaveFocus();
  });

  it('opens pinned on right-click instead of the native menu, and never invokes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke });
    const contextMenu = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });

    await act(() => button.dispatchEvent(contextMenu));

    expect(contextMenu.defaultPrevented).toBe(true);
    await expect.element(destinationGroup()).toBeVisible();
    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);
    expect(invoke).not.toHaveBeenCalled();
  });

  it('changes the destination in place, keeping the card open', async () => {
    const { button } = await renderInvokeButton();

    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await expect.element(destinationGroup()).toBeVisible();
    await act(() => destinationGroup().getByText('Canvas').click());

    expect(harness.destinations).toEqual(['canvas']);
    await expect.element(destinationGroup()).toBeVisible();
    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);
  });

  it('pins a hover preview on a press inside it, so leaving no longer closes it', async () => {
    const { button } = await renderInvokeButton();

    await act(() => userEvent.hover(button));
    await expect.element(destinationGroup()).toBeVisible();
    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);

    await act(() => destinationGroup().getByText('Canvas').click());
    await act(() => userEvent.unhover(destinationGroup()));
    await act(() => userEvent.hover(document.body));

    await expect.element(destinationGroup()).toBeVisible();
    expect(harness.destinations).toEqual(['canvas']);
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

    // The name stays put; the pressed state carries the lock.
    await expect.element(lock).toHaveAttribute('aria-pressed', 'true');
    await expect.element(lock).toHaveFocus();
    expect(button.querySelector('[data-routing-lock-indicator]')).not.toBeNull();
    // The mark is decoration: Invoke keeps its name, and its description says the route is locked.
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await expect
      .element(button)
      .toHaveAccessibleDescription(
        'topbar.invoke.routeDescriptionLocked(widgets.labels.generate, widgets.labels.gallery) topbar.invoke.routeHint'
      );
  });

  it('opens pinned on Alt+ArrowDown too, and keeps both keys from the window’s hotkeys', async () => {
    const windowKeys: string[] = [];
    const recordKey = (event: KeyboardEvent) => windowKeys.push(event.key);
    window.addEventListener('keydown', recordKey);

    try {
      const { button } = await renderInvokeButton();

      button.focus();
      await act(() => userEvent.keyboard('{Alt>}{ArrowDown}{/Alt}'));
      await expect.element(destinationGroup()).toBeVisible();
      expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);
      await act(() => userEvent.keyboard('{Escape}'));
      await expect.element(button).toHaveFocus();
      await act(() => userEvent.keyboard('{ArrowDown}'));
      await expect.element(destinationGroup()).toBeVisible();

      expect(windowKeys.filter((key) => key === 'ArrowDown')).toEqual([]);
    } finally {
      window.removeEventListener('keydown', recordKey);
    }
  });

  it('pins an open preview on ArrowDown and moves focus into it', async () => {
    const { button } = await renderInvokeButton();

    await act(() => userEvent.hover(button));
    await expect.element(destinationGroup()).toBeVisible();
    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));

    expect(button.getAttribute('aria-controls')).toBe(document.querySelector('[role="dialog"]')?.id);
    await expect.poll(() => destinationGroup().element().contains(document.activeElement)).toBe(true);
  });

  it('closes to Invoke on Shift+Tab before the lock, its first control', async () => {
    const { button } = await renderInvokeButton();

    button.focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    // One source shows as text, so focus starts on the selected destination; the lock sits before it.
    await expect.poll(() => (document.activeElement as HTMLInputElement | null)?.value).toBe('gallery');
    await act(() => userEvent.keyboard('{Shift>}{Tab}{/Shift}'));
    await expect.element(page.getByRole('button', { name: 'topbar.routing.lockRouting' })).toHaveFocus();
    await act(() => userEvent.keyboard('{Shift>}{Tab}{/Shift}'));

    await expect.element(button).toHaveFocus();
    await expect.element(destinationGroup()).not.toBeInTheDocument();
  });

  // Focus landing on Invoke unasked would make the next Enter queue a generation.
  it('leaves focus where it is when a card it never held closes', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke });
    const prompt = document.createElement('textarea');
    host?.append(prompt);
    prompt.focus();

    await act(() => userEvent.hover(button));
    await expect.element(destinationGroup()).toBeVisible();
    await act(() => userEvent.unhover(button));
    await expect.element(destinationGroup()).not.toBeInTheDocument();
    expect(document.activeElement).toBe(prompt);

    await act(() => button.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true })));
    await expect.element(destinationGroup()).toBeVisible();
    // The card's dismiss layer arms a frame or so after it shows; repeat Escape until it does.
    await expect
      .poll(async () => {
        await act(() => userEvent.keyboard('{Escape}'));
        return destinationGroup().query();
      })
      .toBeNull();
    expect(document.activeElement).toBe(prompt);

    await act(() => userEvent.keyboard('{Enter}'));
    expect(invoke).not.toHaveBeenCalled();
  });

  it('offers no destination choice for video, which only lands in the Gallery', async () => {
    const { button } = await renderInvokeButton({ invocation: { ...invocation, sourceId: 'video' } });

    await act(() => userEvent.hover(button));

    await expect.element(page.getByText('topbar.routing.destination')).toBeVisible();
    await expect.element(page.getByText('widgets.labels.gallery')).toBeVisible();
    expect(destinationGroup().query()).toBeNull();
  });
});
