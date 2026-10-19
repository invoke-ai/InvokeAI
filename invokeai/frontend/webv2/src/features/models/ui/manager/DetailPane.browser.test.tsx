import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getModelsUiSnapshotForTests, openModelDetail } from '@features/models/ui/uiStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { DetailPane } from './DetailPane';

const api = vi.hoisted(() => ({ deleteModel: vi.fn(() => Promise.resolve()) }));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  deleteModel: api.deleteModel,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, options?: { name?: string }) => `${key}${options?.name ?? ''}` }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const VAE = {
  base: 'sdxl',
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash-vae',
  key: 'vae-1',
  name: 'Model VAE',
  path: '/models/vae.safetensors',
  source: '/models/vae.safetensors',
  source_type: 'path',
  type: 'vae',
} as ModelConfig;

const pathDialog = () =>
  [...document.querySelectorAll('[role="dialog"]')].find((dialog) => dialog.textContent.includes('models.currentPath'));

describe('models DetailPane', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    accountLifecycle.activate('detail-pane-test', ':user:detail-pane-test');
    setModelsSnapshotForTests({ models: [VAE], status: 'loaded' });
    openModelDetail(VAE.key);
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <DetailPane />
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('animates the delete confirmation out after the deleted model leaves the pane', async () => {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="models.actions"]')!));
    await expect.poll(() => document.querySelector('[role="menuitem"][data-value="delete"]')).not.toBeNull();
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="menuitem"][data-value="delete"]')!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
    const dialog = document.querySelector('[role="alertdialog"]')!;
    const confirm = [...dialog.querySelectorAll('button')].find(
      (button) => button.textContent === 'models.deleteModel'
    );

    const frames = closingFrames(
      await recordDialogExit(dialog, async () => {
        await act(() => userEvent.click(confirm!));
        await expect.poll(() => getModelsUiSnapshotForTests().activeModelKey).toBeNull();
      })
    );

    expect(api.deleteModel).toHaveBeenCalledWith(VAE.key, expect.anything());
    expect(frames).not.toHaveLength(0);
    // It still names the model it deleted while it animates out.
    for (const frame of frames) {
      expect(frame.text).toContain('models.deleteBodyModel VAE');
    }
  });

  it('animates the update-path dialog out instead of unmounting it on close', async () => {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="models.updatePath"]')!));
    await expect.poll(() => pathDialog()?.getAttribute('data-state')).toBe('open');

    const frames = closingFrames(
      await recordDialogExit(pathDialog()!, () => act(() => userEvent.keyboard('{Escape}')))
    );
    await expect.poll(() => pathDialog()).toBeUndefined();

    expect(frames).not.toHaveLength(0);
  });
});
