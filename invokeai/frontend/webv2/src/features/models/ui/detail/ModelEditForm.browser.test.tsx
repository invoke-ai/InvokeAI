import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { ModelEditForm } from './ModelEditForm';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(),
  listMissingModels: vi.fn(),
  listModels: vi.fn(),
  updateModel: vi.fn(),
}));

vi.mock('@features/models/data/api', () => api);
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const checkpointMain = {
  base: 'sd-1',
  config_path: 'v1-inference.yaml',
  description: null,
  file_size: 1,
  format: 'checkpoint',
  hash: 'hash',
  key: 'ckpt-main',
  name: 'Checkpoint Main',
  path: '/models/main.ckpt',
  prediction_type: null,
  source: '/models/main.ckpt',
  source_type: 'path',
  type: 'main',
  variant: 'normal',
} as ModelConfig;

const diffusersVae = {
  base: 'sdxl',
  description: null,
  file_size: 1,
  format: 'diffusers',
  hash: 'hash',
  key: 'sdxl-vae',
  name: 'SDXL VAE',
  path: 'sdxl/vae',
  source: 'sdxl/vae',
  source_type: 'path',
  type: 'vae',
} as ModelConfig;

const setInputValue = (input: HTMLInputElement, value: string): void => {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;

  setter.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
};

describe('ModelEditForm', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onCancel: Mock<() => void>;
  let onSaved: Mock<() => void>;

  const mount = async (model: ModelConfig) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelEditForm model={model} onCancel={onCancel} onSaved={onSaved} />
        </ChakraProvider>
      );
    });
  };

  const saveButton = () =>
    [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
      button.textContent.includes('users.saveChanges')
    )!;

  beforeEach(() => {
    api.updateModel.mockReset();
    onCancel = vi.fn();
    onSaved = vi.fn();
    accountLifecycle.activate('model-edit-test-a', ':user:model-edit-test-a');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('offers format, a constrained variant select, and config path for a checkpoint main', async () => {
    await mount(checkpointMain);

    expect(host.querySelector('[aria-label="models.format"]')).not.toBeNull();
    // sd-1 main has a curated variant list, so the field is a select.
    expect(host.querySelector('[aria-label="models.variant"]')).not.toBeNull();
    expect(host.textContent).toContain('models.configPath');
    expect(host.querySelector<HTMLInputElement>('input[value="v1-inference.yaml"]')).not.toBeNull();
  });

  it('hides config path and falls back to free-text variant when the model has neither', async () => {
    await mount(diffusersVae);

    expect(host.textContent).not.toContain('models.configPath');
    // sdxl vae has no variant concept: free-text input, no select trigger.
    expect(host.querySelector('[aria-label="models.variant"]')).toBeNull();
  });

  it('submits format and config_path alongside the identity fields', async () => {
    api.updateModel.mockResolvedValue({ ...checkpointMain, name: 'Renamed' });
    await mount(checkpointMain);

    const nameInput = host.querySelector<HTMLInputElement>('input[value="Checkpoint Main"]')!;

    await act(async () => {
      setInputValue(nameInput, 'Renamed');
      await Promise.resolve();
    });

    await act(async () => {
      saveButton().click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(
      'ckpt-main',
      expect.objectContaining({
        base: 'sd-1',
        config_path: 'v1-inference.yaml',
        format: 'checkpoint',
        name: 'Renamed',
        type: 'main',
        variant: 'normal',
      }),
      expect.anything()
    );
    expect(onSaved).toHaveBeenCalledOnce();
  });

  it('nulls a cleared config path but never sends one for models without the field', async () => {
    api.updateModel.mockResolvedValue(checkpointMain);
    await mount(checkpointMain);

    const configInput = host.querySelector<HTMLInputElement>('input[value="v1-inference.yaml"]')!;

    await act(async () => {
      setInputValue(configInput, '');
      await Promise.resolve();
    });

    await act(async () => {
      saveButton().click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(
      'ckpt-main',
      expect.objectContaining({ config_path: null }),
      expect.anything()
    );

    api.updateModel.mockClear();
    api.updateModel.mockResolvedValue(diffusersVae);
    await mount(diffusersVae);

    await act(async () => {
      saveButton().click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledOnce();
    expect(api.updateModel.mock.calls[0]![1]).not.toHaveProperty('config_path');
  });
});
