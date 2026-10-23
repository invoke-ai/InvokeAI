import type { InvocationRoute } from '@workbench/invocationContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { InvocationState } from './useInvocationState';
import type * as UseTopbarShortcutModule from './useTopbarShortcut';

// Verify progress returns after hover ends unless focus-visible remains; mouse click-focus must not pin the play
// icon. Mock queue plumbing to isolate that gate.
const harness = vi.hoisted(() => ({
  bindings: {} as Record<string, { aria: string; display: string; parts: string[] }>,
  progress: { activeItemIndex: 1, completedItemCount: 0, message: '', percentage: 0.42 as number | null },
  summary: { current: 1, remaining: 1, runningQueueItemId: 'item-1', total: 1 },
}));

vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({ progress: harness.progress, queueItems: [], summary: harness.summary }),
}));
vi.mock('@workbench/invocation', () => ({ getDestinationLabel: () => 'Gallery' }));
vi.mock('./useTopbarShortcut', async (importOriginal) => ({
  ...(await importOriginal<typeof UseTopbarShortcutModule>()),
  useTopbarShortcutBinding: (commandId: string) => harness.bindings[commandId] ?? null,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { shortcut?: string }) => (options?.shortcut ? `${key}: ${options.shortcut}` : key),
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
  blockingReasons: [],
  invocation,
  invoke: vi.fn(() => Promise.resolve()),
  isPreparing: false,
  isValid: true,
  placedTypeIds: new Set<never>(),
  promptExpansion: { count: 1, error: null, isDynamic: false, isError: false, isLoading: false, prompts: [] },
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

const hasProgressRing = (button: HTMLButtonElement): boolean => button.querySelector('[role="progressbar"]') !== null;
const hasSpinner = (button: HTMLButtonElement): boolean => button.querySelector('.chakra-spinner') !== null;

beforeEach(() => {
  harness.bindings = {};
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
    const { button } = await renderInvokeButton({ invoke, isPreparing: true });

    expect(button.getAttribute('aria-disabled')).toBe('true');
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await expect.element(button).toHaveAccessibleDescription('topbar.invoke.preparing');
    expect(hasSpinner(button)).toBe(true);

    await act(() => button.click());

    expect(invoke).not.toHaveBeenCalled();
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

  it('ignores both clicks while Invoke is blocked', async () => {
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({
      blockingReasons: ['The backend is disconnected.'],
      invoke,
      isValid: false,
    });

    // The visible label names it (WCAG 2.5.3); the blocking reason describes it.
    expect(button.getAttribute('aria-disabled')).toBe('true');
    await expect.element(button).toHaveAccessibleName('topbar.invoke.invoke');
    await expect.element(button).toHaveAccessibleDescription('topbar.invoke.unavailable');

    // Dispatched: a blocked Invoke is not actionable for a pointer driver.
    await act(() => button.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    await act(() => button.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true })));

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
