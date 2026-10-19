import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { MaintenanceMenu } from './MaintenanceMenu';

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getOrphanedModels: vi.fn(() => Promise.resolve([])),
}));
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
});
