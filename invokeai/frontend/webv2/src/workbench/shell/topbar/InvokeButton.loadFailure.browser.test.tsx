import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { InvocationState } from './useInvocationState';

const notify = vi.hoisted(() => ({ error: vi.fn() }));

vi.mock('@workbench/queue-integration/useActiveQueueProgress', () => ({
  useActiveQueueProgress: () => ({ progress: null, queueItems: [], summary: { total: 0 } }),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));
vi.mock('@workbench/hotkeys/useShortcutBinding', () => ({ useShortcutBinding: () => null }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
// The card's module fails as a dropped connection would. Its own file: a page loads that module once.
vi.mock('./InvokePopoverBody', () => ({
  get InvokePopoverBody() {
    throw new TypeError('Failed to fetch dynamically imported module');
  },
}));

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

it('answers a click whose card fails to load, and leaves the button ready to try again', async () => {
  document.body.append(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <InvokeButton state={BLOCKED_STATE} />
      </ChakraProvider>
    )
  );
  const button = host.querySelector<HTMLButtonElement>('button[data-blocked]')!;

  await act(() => userEvent.click(button));
  await expect.poll(() => notify.error.mock.calls).toEqual([['topbar.invoke.cardLoadFailed']]);
  expect(button.hasAttribute('aria-busy')).toBe(false);
  expect(button.getAttribute('aria-expanded')).toBe('false');
  expect(document.querySelector('[role="dialog"]')).toBeNull();
});
