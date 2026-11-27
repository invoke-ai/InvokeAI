import { isModalPresent } from '@platform/ui/modalPresence';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

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
  expect(isModalPresent()).toBe(true);

  await act(async () => {
    chunk.arrive();
    await chunk.arrived;
  });
  await expect.poll(() => host.querySelector('[data-testid="paste"]')).not.toBeNull();
  expect(isModalPresent()).toBe(false);
});
