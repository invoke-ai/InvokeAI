import type { InvocationRoute } from '@workbench/invocationContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { InvocationState } from './useInvocationState';
import type * as UseTopbarShortcutModule from './useTopbarShortcut';

// Isolates the regression this file exists to catch: the icon slot must swap
// back to progress once the pointer leaves AND the button does not hold
// `:focus-visible` — a plain mouse click leaves it focused (browsers focus a
// button on mousedown) but must not count, or every mouse-invoked batch would
// show a play glyph it cannot act on. Mocking the shared hook keeps the test
// about that gating, not about queue-summary plumbing (which
// `useActiveQueueProgress` and its call sites already cover elsewhere).
const harness = vi.hoisted(() => ({
  progress: { activeItemIndex: 1, completedItemCount: 0, message: '', percentage: 0.42 },
  summary: { current: 1, remaining: 1, runningQueueItemId: 'item-1', total: 1 },
}));

vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({ progress: harness.progress, queueItems: [], summary: harness.summary }),
}));
vi.mock('@workbench/invocation', () => ({ getDestinationLabel: () => 'Gallery' }));
vi.mock('./useTopbarShortcut', async (importOriginal) => ({
  ...(await importOriginal<typeof UseTopbarShortcutModule>()),
  useTopbarShortcutBinding: () => null,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

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

beforeEach(() => {
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

  it('acknowledges canvas preparation immediately and ignores a second click', async () => {
    harness.summary.total = 0;
    const invoke = vi.fn(() => Promise.resolve());
    const { button } = await renderInvokeButton({ invoke, isPreparing: true });

    expect(button.getAttribute('aria-disabled')).toBe('true');
    expect(button.getAttribute('aria-label')).toBe('topbar.invoke.preparing');
    expect(hasProgressRing(button)).toBe(true);

    await act(() => button.click());

    expect(invoke).not.toHaveBeenCalled();
  });
});
