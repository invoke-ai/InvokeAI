import type * as settingsStoreModule from '@workbench/settings/store';

import { isModalPresent } from '@platform/ui/modalPresence';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

/** Holds the palette's module back, as a slow connection does on its first open. */
const chunk = vi.hoisted(() => {
  let arrive = () => {};
  const arrived = new Promise<void>((resolve) => {
    arrive = resolve;
  });

  return { arrive, arrived };
});

vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<typeof settingsStoreModule>()),
  useWorkbenchPreferences: () => ({}),
}));

vi.mock('./WorkbenchCommandPaletteDialog', async () => {
  await chunk.arrived;

  return { default: ({ isOpen }: { isOpen: boolean }) => <div data-open={String(isOpen)} data-testid="palette" /> };
});

import { closeCommandPalette, openCommandPalette } from './paletteStore';
import { WorkbenchCommandPalette } from './WorkbenchCommandPalette';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const host = document.createElement('div');
const root = createRoot(host);

afterEach(async () => {
  await act(() => {
    closeCommandPalette();
    root.unmount();
  });
  host.remove();
});

it('suspends shortcuts from the moment the palette opens, while its module is still loading', async () => {
  document.body.append(host);
  await act(() => root.render(<WorkbenchCommandPalette />));

  await act(() => openCommandPalette());
  expect(host.querySelector('[data-testid="palette"]')).toBeNull();
  expect(isModalPresent()).toBe(true);

  await act(async () => {
    chunk.arrive();
    await chunk.arrived;
  });
  await expect.poll(() => host.querySelector('[data-testid="palette"]')).not.toBeNull();
  // The stand-in steps aside for the loaded dialog, which this test replaces with one that announces nothing.
  expect(isModalPresent()).toBe(false);
});
