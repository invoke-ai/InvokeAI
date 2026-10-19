import { ChakraProvider } from '@chakra-ui/react';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

vi.mock('@features/queue/publicApi', () => ({
  createProductionQueueRealtimeRuntime: () => ({ dispose: vi.fn(), start: vi.fn() }),
}));
vi.mock('@features/queue/data/generationDevicesStore', () => ({ refreshGenerationDevices: vi.fn() }));

import { clearQueueConfirmation, requestQueueConfirmation } from './queueConfirmationStore';
import { QueueDataRuntime } from './QueueDataRuntime';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueueDataRuntime />
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  clearQueueConfirmation();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('animates a queue confirmation out with its text instead of unmounting it on close', async () => {
  await act(() =>
    requestQueueConfirmation({
      body: 'Cancel every item?',
      confirmLabel: 'Cancel all',
      onConfirm: vi.fn(),
      title: 'Clear',
    })
  );
  await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
  const dialog = document.querySelector('[role="alertdialog"]')!;

  const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));

  // An unmounted dialog never reaches its closed state, so it cannot animate out; a retained one does, then leaves.
  expect(frames).not.toHaveLength(0);
  for (const frame of frames) {
    expect(frame.text).toContain('Cancel every item?');
  }
  await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
});
