import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { MaintenanceMenu } from './MaintenanceMenu';

const dependencies = vi.hoisted(() => ({
  emptyModelCache: vi.fn(),
  notify: { error: vi.fn(), info: vi.fn(), success: vi.fn() },
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  emptyModelCache: dependencies.emptyModelCache,
  getOrphanedModels: vi.fn(() => Promise.resolve([])),
}));
vi.mock('@features/models/ui/useModelsNotify', () => ({ useNotify: () => dependencies.notify }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('MaintenanceMenu', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    accountLifecycle.activate('maintenance-menu-test', ':user:maintenance-menu-test');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <MaintenanceMenu />
        </ChakraProvider>
      )
    );
  });

  beforeEach(() => {
    dependencies.emptyModelCache.mockReset();
    dependencies.notify.error.mockReset();
    dependencies.notify.success.mockReset();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('animates the orphaned-models dialog out instead of unmounting it on close', async () => {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="models.libraryMaintenance"]')!));
    await expect.poll(() => document.querySelector('[role="menuitem"][data-value="sync"]')).not.toBeNull();
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="menuitem"][data-value="sync"]')!));
    await expect.poll(() => document.querySelector('[role="dialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="dialog"]')!;
    await expect.poll(() => dialog.textContent).toContain('models.noOrphaned');

    const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();

    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain('models.noOrphaned');
    }
  });

  it.each([
    { refused: true, error: ['models.vramNotReleased', 'models.vramNotReleasedDescription'], success: null },
    { refused: false, error: null, success: ['models.cacheEmptied'] },
  ])('reports an emptied cache with refused VRAM release = $refused', async ({ refused, error, success }) => {
    dependencies.emptyModelCache.mockResolvedValue({
      bytes_freed: 1,
      models_cleared: 1,
      vram_release_refused: refused,
    });

    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="models.libraryMaintenance"]')!));
    await expect.poll(() => document.querySelector('[role="menuitem"][data-value="empty-cache"]')).not.toBeNull();
    await act(() =>
      userEvent.click(document.querySelector<HTMLElement>('[role="menuitem"][data-value="empty-cache"]')!)
    );
    const findConfirm = () =>
      [...document.querySelectorAll<HTMLElement>('button')].find(
        (button) => button.textContent?.trim() === 'models.emptyCache'
      );
    await expect.poll(findConfirm).toBeDefined();
    const confirm = findConfirm()!;
    await act(() => userEvent.click(confirm));

    await expect.poll(() => dependencies.emptyModelCache.mock.calls.length).toBe(1);
    await expect
      .poll(() => dependencies.notify.error.mock.calls.length + dependencies.notify.success.mock.calls.length)
      .toBe(1);
    expect(dependencies.notify.error.mock.calls[0] ?? null).toEqual(error);
    expect(dependencies.notify.success.mock.calls[0] ?? null).toEqual(success);
  });
});
