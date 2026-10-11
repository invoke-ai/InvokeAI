import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelDraft } from '@features/models/ui/modelDraftsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ExternalModelSettingsModel } from './externalModelFields';

import { ExternalModelSection } from './ExternalModelSection';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(() => Promise.resolve('')),
  listMissingModels: vi.fn(() => Promise.resolve([])),
  listModels: vi.fn(() => Promise.resolve([])),
  updateModel: vi.fn(),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  ...api,
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const gptImage: ExternalModelSettingsModel = {
  capabilities: {
    allowed_aspect_ratios: ['1:1', '3:2', '2:3'],
    max_images_per_request: 10,
    modes: ['txt2img', 'img2img'],
    supports_reference_images: true,
  },
  default_settings: { height: 1024, num_images: 1, width: 1024 },
  key: 'gpt-image-1',
  name: 'GPT Image 1',
  provider_id: 'openai',
  provider_model_id: 'gpt-image-1',
  type: 'external_image_generator',
};

const setInputValue = (input: HTMLInputElement, value: string): void => {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;

  setter.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
};

describe('ExternalModelSection', () => {
  let host: HTMLDivElement;
  let root: Root;
  const onError = vi.fn();
  const onSaved = vi.fn();

  const render = async (model: ExternalModelSettingsModel) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ExternalModelSection model={model} onError={onError} onSaved={onSaved} />
        </ChakraProvider>
      );
    });
  };

  const input = (label: string) => host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`)!;
  const checkbox = (label: string) =>
    [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((candidate) =>
      candidate.closest('label')?.textContent?.includes(label)
    )!;
  const button = (label: string) =>
    [...host.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => candidate.textContent === label)!;
  // Number inputs only take edits while focused, as a user's would be.
  const type = async (element: HTMLInputElement, value: string) => {
    await act(async () => {
      element.focus();
      setInputValue(element, value);
      element.blur();
      await Promise.resolve();
    });
  };
  const click = async (element: HTMLElement) => {
    await act(async () => {
      element.click();
      await Promise.resolve();
    });
  };

  beforeEach(() => {
    api.updateModel.mockReset();
    onError.mockReset();
    onSaved.mockReset();
    accountLifecycle.activate('external-section-test', ':user:external-section-test');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('shows the record and keeps Save disabled until something changes', async () => {
    await render(gptImage);

    expect(host.querySelector<HTMLInputElement>('input[value="gpt-image-1"]')).not.toBeNull();
    expect(checkbox('models.externalModes.txt2img').checked).toBe(true);
    expect(checkbox('models.externalModes.img2img').checked).toBe(true);
    expect(checkbox('models.externalModes.inpaint').checked).toBe(false);
    expect(checkbox('models.externalFields.supportsReferenceImages').checked).toBe(true);
    expect(checkbox('models.externalFields.supportsSeed').checked).toBe(false);
    expect(input('models.externalFields.maxImagesPerRequest').value).toBe('10');
    expect(input('models.externalFields.maxReferenceImages').value).toBe('');
    expect(input('models.externalFields.defaultWidth').value).toBe('1024');
    expect(button('models.saveExternalSettings').disabled).toBe(true);
  });

  it('saves the whole provider record through the model update transport and settles the draft', async () => {
    api.updateModel.mockImplementation((_key: string, changes: Record<string, unknown>) =>
      Promise.resolve({ ...gptImage, ...changes })
    );
    await render(gptImage);

    await click(checkbox('models.externalModes.inpaint'));
    await click(checkbox('models.externalFields.supportsSeed'));
    await type(input('models.externalFields.maxReferenceImages'), '4');
    await type(input('models.externalFields.defaultNumImages'), '');
    const providerModelInput = host.querySelector<HTMLInputElement>('input[value="gpt-image-1"]')!;
    await type(providerModelInput, 'gpt-image-1.5');

    expect(host.textContent).toContain('models.unsavedChanges');
    await click(button('models.saveExternalSettings'));

    expect(api.updateModel).toHaveBeenCalledWith(
      'gpt-image-1',
      {
        capabilities: {
          allowed_aspect_ratios: ['1:1', '3:2', '2:3'],
          max_image_size: null,
          max_images_per_request: 10,
          max_reference_images: 4,
          modes: ['txt2img', 'img2img', 'inpaint'],
          supports_negative_prompt: true,
          supports_reference_images: true,
          supports_seed: true,
        },
        default_settings: { height: 1024, num_images: null, width: 1024 },
        provider_id: 'openai',
        provider_model_id: 'gpt-image-1.5',
      },
      expect.anything()
    );
    expect(onSaved).toHaveBeenCalledOnce();
    expect(getModelDraft('gpt-image-1')?.external).toBeUndefined();
  });

  it('refuses to save an invalid record and reports why without calling the backend', async () => {
    await render(gptImage);

    await click(checkbox('models.externalModes.txt2img'));
    await click(checkbox('models.externalModes.img2img'));
    await click(button('models.saveExternalSettings'));

    expect(host.querySelector('[role="alert"]')?.textContent).toBe('Choose at least one supported mode.');
    expect(api.updateModel).not.toHaveBeenCalled();
    // Reset discards the draft and the error together.
    await click(button('common.reset'));
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(checkbox('models.externalModes.txt2img').checked).toBe(true);
  });

  it('surfaces a failed save and keeps the edits for another attempt', async () => {
    api.updateModel.mockRejectedValue(new Error('Model not found'));
    await render(gptImage);

    await click(checkbox('models.externalFields.supportsSeed'));
    await click(button('models.saveExternalSettings'));

    expect(onError).toHaveBeenCalledWith('Model not found');
    expect(checkbox('models.externalFields.supportsSeed').checked).toBe(true);
    expect(button('models.saveExternalSettings').disabled).toBe(false);
  });
});
