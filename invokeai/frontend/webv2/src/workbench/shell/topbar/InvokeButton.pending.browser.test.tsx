import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { InvocationState } from './useInvocationState';

const chunk = vi.hoisted(() => {
  let arrive = () => {};
  const arrived = new Promise<void>((resolve) => {
    arrive = resolve;
  });

  return { arrive, arrived, requests: 0 };
});

vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({ progress: null, queueItems: [], summary: { total: 0 } }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: () => {} }) }));
vi.mock('@workbench/hotkeys/useShortcutBinding', () => ({ useShortcutBinding: () => null }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('./InvokePopoverBody', async () => {
  chunk.requests += 1;
  await chunk.arrived;

  return {
    InvokePopoverBody: () => (
      <button data-issue-action="" type="button">
        Show layer
      </button>
    ),
  };
});

import { InvokeButton } from './InvokeButton';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const BLOCKED_STATE = {
  invocation: { destination: 'gallery', destinationLocked: false, sourceId: 'generate', sourceLocked: false },
  invoke: () => Promise.resolve(),
  isPreparing: false,
  isValid: false,
  issues: [{ message: 'Choose a model.', target: { kind: 'connection' } }],
} as unknown as InvocationState;

const host = document.createElement('div');
const root = createRoot(host);

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const waitPastPreviewDelay = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        window.setTimeout(resolve, 400);
      })
  );

it('holds opens made before the card loads: a blur drops a click, Enter pins, and a pointer re-entry keeps it pinned', async () => {
  document.body.append(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <InvokeButton state={BLOCKED_STATE} />
        <InvokeButton state={BLOCKED_STATE} />
      </ChakraProvider>
    )
  );
  const [left, right] = host.querySelectorAll<HTMLButtonElement>('button[data-blocked]');

  await act(() => userEvent.click(left!));
  expect(chunk.requests).toBe(1);
  expect(left!.getAttribute('aria-busy')).toBe('true');
  expect(left!.querySelector('.chakra-spinner')).not.toBeNull();
  expect(document.querySelector('[role="dialog"]')).toBeNull();

  await act(() => userEvent.tab());
  expect(document.activeElement).toBe(right);
  expect(left!.hasAttribute('aria-busy')).toBe(false);

  await act(() => userEvent.keyboard('{Enter}'));
  expect(right!.getAttribute('aria-busy')).toBe('true');
  await act(() => userEvent.hover(right!));
  await act(() => userEvent.unhover(right!));
  await act(() => userEvent.hover(right!));
  await waitPastPreviewDelay();
  expect(right!.getAttribute('aria-expanded')).toBe('false');

  await act(async () => {
    chunk.arrive();
    await chunk.arrived;
  });

  await expect.element(page.getByRole('button', { name: 'Show layer' })).toHaveFocus();
  expect(right!.getAttribute('aria-expanded')).toBe('true');
  expect(right!.hasAttribute('aria-busy')).toBe(false);
  expect(left!.getAttribute('aria-expanded')).toBe('false');
  expect(document.querySelectorAll('[role="dialog"]')).toHaveLength(1);
});
