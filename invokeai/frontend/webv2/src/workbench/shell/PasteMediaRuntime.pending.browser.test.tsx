import { isModalPresent } from '@platform/ui/modalPresence';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

/** Holds the dialog's module back, as a slow connection does on the first paste. */
const chunk = vi.hoisted(() => {
  let arrive = () => {};
  const arrived = new Promise<void>((resolve) => {
    arrive = resolve;
  });

  return { arrive, arrived };
});

vi.mock('./PasteMediaDialog', async () => {
  await chunk.arrived;

  return {
    PasteMediaDialog: ({ isOpen }: { isOpen: boolean }) => <div data-open={String(isOpen)} data-testid="paste" />,
  };
});

import { PasteMediaRuntime } from './PasteMediaRuntime';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const host = document.createElement('div');
const root = createRoot(host);

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

it('suspends shortcuts from the paste that opens the dialog, while its module is still loading', async () => {
  document.body.append(host);
  await act(() => root.render(<PasteMediaRuntime />));

  const clipboardData = new DataTransfer();
  clipboardData.items.add(new File(['x'], 'shot.png', { type: 'image/png' }));
  await act(() =>
    document.body.dispatchEvent(new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData }))
  );
  expect(host.querySelector('[data-testid="paste"]')).toBeNull();
  // Delete or Ctrl+Enter must not reach the gallery behind the dialog that is on its way.
  expect(isModalPresent()).toBe(true);

  await act(async () => {
    chunk.arrive();
    await chunk.arrived;
  });
  await expect.poll(() => host.querySelector('[data-testid="paste"]')).not.toBeNull();
  // The stand-in steps aside for the loaded dialog, which this test replaces with one that announces nothing.
  expect(isModalPresent()).toBe(false);
});
