import type * as settingsStoreModule from '@workbench/settings/store';

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const workbenchDialog = vi.hoisted(() => ({ mounts: 0, onExitComplete: null as (() => void) | null }));

vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<typeof settingsStoreModule>()),
  useWorkbenchPreferences: () => ({}),
}));

vi.mock('./WorkbenchCommandPaletteDialog', async () => {
  const { useMountEffect } = await import('@platform/react/useMountEffect');

  const WorkbenchCommandPaletteDialog = ({
    isOpen,
    onExitComplete,
  }: {
    isOpen: boolean;
    onExitComplete: () => void;
  }) => {
    useMountEffect(() => {
      workbenchDialog.mounts += 1;
      workbenchDialog.onExitComplete = onExitComplete;
    });

    return <div data-open={String(isOpen)} data-testid="workbench-palette" />;
  };

  return { default: WorkbenchCommandPaletteDialog };
});

import { closeCommandPalette, openCommandPalette } from './paletteStore';
import { WorkbenchCommandPalette } from './WorkbenchCommandPalette';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const palette = () => document.querySelector<HTMLElement>('[data-testid="workbench-palette"]');

beforeEach(async () => {
  workbenchDialog.mounts = 0;
  workbenchDialog.onExitComplete = null;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root?.render(<WorkbenchCommandPalette />));
});

afterEach(async () => {
  await act(() => {
    closeCommandPalette();
    root?.unmount();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('WorkbenchCommandPalette host', () => {
  it('keeps the lazy dialog mounted while it animates closed', async () => {
    await act(() => openCommandPalette());
    await expect.poll(() => palette()?.dataset.open).toBe('true');

    await act(() => closeCommandPalette());
    expect(palette()?.dataset.open).toBe('false');

    await act(() => workbenchDialog.onExitComplete?.());
    expect(palette()).toBeNull();

    // Each open starts a fresh palette rather than reusing the one that closed.
    await act(() => openCommandPalette());
    await expect.poll(() => palette()?.dataset.open).toBe('true');
    expect(workbenchDialog.mounts).toBe(2);
  });

  it('starts a fresh palette when reopened before the previous one finished closing', async () => {
    await act(() => openCommandPalette());
    await expect.poll(() => palette()?.dataset.open).toBe('true');
    await act(() => closeCommandPalette());

    await act(() => openCommandPalette());
    expect(palette()?.dataset.open).toBe('true');
    expect(workbenchDialog.mounts).toBe(2);
  });
});
