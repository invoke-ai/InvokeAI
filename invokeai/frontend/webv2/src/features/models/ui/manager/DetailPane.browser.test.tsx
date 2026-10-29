import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelsSnapshot, setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { type ModelsUiAdapter, ModelsUiProvider, NO_SCHEDULER_POLICY } from '@features/models/ui/ModelsUiContext';
import { getModelsUiSnapshotForTests, openModelDetail } from '@features/models/ui/uiStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { DetailPane } from './DetailPane';

const api = vi.hoisted(() => ({
  deleteModel: vi.fn(() => Promise.resolve()),
  reidentifyModel: vi.fn<(key: string) => Promise<ModelConfig>>(),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  deleteModel: api.deleteModel,
  reidentifyModel: api.reidentifyModel,
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string; scale?: number }) =>
      `${key}${options?.name ?? ''}${options?.scale ?? ''}`,
  }),
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

describe('models DetailPane for a model installed before the probe recorded a fact', () => {
  const UNIDENTIFIED_LLLITE = {
    ...VAE,
    base: 'anima',
    cond_in_channels: null,
    hash: 'hash-lllite',
    key: 'lllite-1',
    name: 'Anima LLLite',
    path: '/models/lllite.safetensors',
    source: '/models/lllite.safetensors',
    type: 'controlnet',
  } as ModelConfig;
  const adapter = (canManageModels: boolean): ModelsUiAdapter => ({
    canManageModels,
    enableModelDescriptions: true,
    installInPlace: true,
    isProjectActive: () => true,
    managerProjectId: null,
    schedulerPolicy: NO_SCHEDULER_POLICY,
    setInstallInPlace: () => undefined,
  });
  let host: HTMLDivElement;
  let root: Root;

  const UNSCALED_UPSCALER = {
    ...VAE,
    base: 'any',
    hash: 'hash-upscaler',
    key: 'upscaler-1',
    name: 'RealESRGAN x4',
    path: '/models/realesrgan.safetensors',
    scale: null,
    source: '/models/realesrgan.safetensors',
    type: 'spandrel_image_to_image',
  } as ModelConfig;
  const render = async (canManageModels: boolean, model: ModelConfig = UNIDENTIFIED_LLLITE) => {
    accountLifecycle.activate('detail-pane-lllite', ':user:detail-pane-lllite');
    setModelsSnapshotForTests({ models: [model], status: 'loaded' });
    openModelDetail(model.key);
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={adapter(canManageModels)}>
            <DetailPane />
          </ModelsUiProvider>
        </ChakraProvider>
      )
    );
  };
  const reidentifyButton = () =>
    [...host.querySelectorAll('button')].find((button) => button.textContent === 'models.reidentify');
  const openConfirm = async () => {
    await act(() => userEvent.click(reidentifyButton()!));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')?.getAttribute('data-state')).toBe('open');
    return document.querySelector<HTMLElement>('[role="alertdialog"]')!;
  };
  const dialogButton = (dialog: HTMLElement, label: string) =>
    [...dialog.querySelectorAll('button')].find((button) => button.textContent === label)!;

  beforeEach(() => {
    api.reidentifyModel.mockReset();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('explains the unknown type and re-identifies only after a confirm that names the reset', async () => {
    api.reidentifyModel.mockResolvedValue({ ...UNIDENTIFIED_LLLITE, cond_in_channels: 3 });
    await render(true);

    expect(host.textContent).toContain('models.adapterTypeUnknown');
    expect(host.textContent).not.toContain('models.reidentifyAskAdmin');

    const cancelled = await openConfirm();
    expect(cancelled.textContent).toContain('models.reidentifyConfirmBody');
    await act(() => userEvent.click(dialogButton(cancelled, 'Cancel')));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(api.reidentifyModel).not.toHaveBeenCalled();
    // Cancel returns focus to the button that opened the dialog.
    expect(document.activeElement).toBe(reidentifyButton());

    const confirmed = await openConfirm();
    await act(() => userEvent.click(dialogButton(confirmed, 'models.reidentify')));

    await expect.poll(() => getModelsSnapshot().modelsByKey.get(UNIDENTIFIED_LLLITE.key)?.cond_in_channels).toBe(3);
    expect(api.reidentifyModel).toHaveBeenCalledExactlyOnceWith(UNIDENTIFIED_LLLITE.key, expect.anything());
    expect(host.textContent).not.toContain('models.adapterTypeUnknown');
    // The notice and its button are gone; focus lands on the model's actions rather than the page body.
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).toBeNull();
    await expect.poll(() => document.activeElement).toBe(host.querySelector('[aria-label="models.actions"]'));
  });

  it('tells users who cannot manage models to ask an administrator, without the button', async () => {
    await render(false);

    expect(host.textContent).toContain('models.adapterTypeUnknown');
    expect(host.textContent).toContain('models.reidentifyAskAdmin');
    expect(reidentifyButton()).toBeUndefined();
  });

  it('shows an upscaler’s recorded native scale, with no notice', async () => {
    await render(true, { ...UNSCALED_UPSCALER, scale: 4 } as ModelConfig);

    expect(host.textContent).toContain('models.nativeScale');
    expect(host.textContent).toContain('models.nativeScaleValue4');
    expect(host.textContent).not.toContain('models.nativeScaleUnknown');
    expect(reidentifyButton()).toBeUndefined();
  });

  it('explains an unrecorded native scale and records it on a confirmed re-identify', async () => {
    api.reidentifyModel.mockResolvedValue({ ...UNSCALED_UPSCALER, scale: 4 } as ModelConfig);
    await render(true, UNSCALED_UPSCALER);

    expect(host.textContent).toContain('models.nativeScaleUnknown');
    expect(host.textContent).not.toContain('models.nativeScaleValue');

    const dialog = await openConfirm();
    expect(dialog.textContent).toContain('models.reidentifyConfirmBody');
    await act(() => userEvent.click(dialogButton(dialog, 'models.reidentify')));

    await expect.poll(() => getModelsSnapshot().modelsByKey.get(UNSCALED_UPSCALER.key)?.scale).toBe(4);
    expect(api.reidentifyModel).toHaveBeenCalledExactlyOnceWith(UNSCALED_UPSCALER.key, expect.anything());
    expect(host.textContent).not.toContain('models.nativeScaleUnknown');
    expect(host.textContent).toContain('models.nativeScaleValue4');
  });

  it('tells a user who cannot manage models to ask about an unrecorded scale', async () => {
    await render(false, UNSCALED_UPSCALER);

    expect(host.textContent).toContain('models.nativeScaleUnknown');
    expect(host.textContent).toContain('models.reidentifyAskAdmin');
    expect(reidentifyButton()).toBeUndefined();
  });
});
