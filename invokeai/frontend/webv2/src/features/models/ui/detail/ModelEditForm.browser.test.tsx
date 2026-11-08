import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelTypeLabel, getModelVariantLabel } from '@features/models/core/taxonomy';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
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

/** Each select trigger's accessible name, resolved through its aria-labelledby references. */
const selectTriggerNames = (host: HTMLElement): string[] =>
  [...host.querySelectorAll('[data-scope="select"][data-part="trigger"]')].map((trigger) =>
    (trigger.getAttribute('aria-labelledby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent?.trim() ?? '')
      .join(' ')
      .trim()
  );

const selectTrigger = (host: HTMLElement, name: string): HTMLButtonElement => {
  const triggers = [...host.querySelectorAll<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]')];

  return triggers[selectTriggerNames(host).indexOf(name)]!;
};

const option = (label: string): HTMLElement =>
  [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((item) => item.textContent === label)!;

const pick = async (host: HTMLElement, name: string, label: string): Promise<void> => {
  await act(async () => {
    selectTrigger(host, name).click();
    await Promise.resolve();
  });
  await act(async () => {
    option(label).click();
    await Promise.resolve();
  });
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

    // Each select trigger is named by its surrounding field's label.
    expect(selectTriggerNames(host)).toContain('models.format');
    // sd-1 main has a curated variant list, so the field is a select.
    expect(selectTriggerNames(host)).toContain('models.variant');
    expect(host.textContent).toContain('models.configPath');
    expect(host.querySelector<HTMLInputElement>('input[value="v1-inference.yaml"]')).not.toBeNull();
  });

  it('hides config path, variant and prediction type when the config has none of them', async () => {
    await mount(diffusersVae);

    expect(host.textContent).not.toContain('models.configPath');
    expect(host.textContent).not.toContain('models.variant');
    expect(host.textContent).not.toContain('models.predictionType');
  });

  it('falls back to a free-text variant when the config has one but no curated choices', async () => {
    await mount({ ...diffusersVae, variant: null } as ModelConfig);

    expect(host.textContent).toContain('models.variant');
    expect(selectTriggerNames(host)).not.toContain('models.variant');
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
    const body = api.updateModel.mock.calls[0]![1] as object;
    expect(body).not.toHaveProperty('config_path');
    expect(body).not.toHaveProperty('prediction_type');
    expect(body).not.toHaveProperty('variant');
  });

  it('offers variant and prediction type when reclassifying into a main model, and sends what was picked', async () => {
    api.updateModel.mockResolvedValue({ ...diffusersVae, prediction_type: null, type: 'main', variant: 'inpaint' });
    await mount(diffusersVae);

    await pick(host, 'models.type', getModelTypeLabel('main'));
    expect(selectTriggerNames(host)).toEqual(expect.arrayContaining(['models.variant', 'models.predictionType']));

    await pick(host, 'models.variant', getModelVariantLabel('inpaint'));
    await act(async () => {
      saveButton().click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(
      'sdxl-vae',
      expect.objectContaining({ prediction_type: null, type: 'main', variant: 'inpaint' }),
      expect.anything()
    );
  });

  it("shows the backend's refusal as readable text", async () => {
    api.updateModel.mockRejectedValue(
      new ApiError(JSON.stringify({ detail: "Model 'Checkpoint Main' does not support variant" }), 409)
    );
    await mount(checkpointMain);

    await act(async () => {
      saveButton().click();
      await Promise.resolve();
    });

    expect(host.querySelector('[role="alert"]')?.textContent).toBe("Model 'Checkpoint Main' does not support variant");
    expect(onSaved).not.toHaveBeenCalled();
  });
});
