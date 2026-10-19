import type { ExternalModelSettingsModel } from '@features/models/core/externalModelFields';
import type { ModelConfig } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { toExternalModelValues } from '@features/models/core/externalModelFields';
import { setModelsSnapshotForTests, useModelsSelector } from '@features/models/data/modelsStore';
import { recordModelDraftFields, getModelDraft } from '@features/models/ui/modelDraftsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ExternalModelSection } from './ExternalModelSection';

const api = vi.hoisted(() => ({
  getExternalProviderConfigs: vi.fn(() =>
    Promise.resolve([
      { api_key_configured: true, base_url: null, provider_id: 'openai' },
      { api_key_configured: false, base_url: null, provider_id: 'gemini' },
    ])
  ),
  getModelsDir: vi.fn(() => Promise.resolve('')),
  getStarterModels: vi.fn(() =>
    Promise.resolve({
      starter_bundles: {},
      starter_models: [{ source: 'external://gemini/gemini-2.5-flash-image' }],
    })
  ),
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
  source: 'external://openai/gpt-image-1',
  type: 'external_image_generator',
};

// Reads the record from the library store, as the model detail does, so a save's reply reaches the form.
const LibraryBackedSection = ({ onError, onSaved }: { onError: (message: string) => void; onSaved: () => void }) => {
  const model = useModelsSelector((snapshot) => snapshot.models.find((candidate) => candidate.key === gptImage.key));

  return model ? <ExternalModelSection model={model} onError={onError} onSaved={onSaved} /> : null;
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

  // The editor is read-only until the starter catalog has answered, so every render waits for it.
  const render = async (model: ExternalModelSettingsModel) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ExternalModelSection model={model} onError={onError} onSaved={onSaved} />
        </ChakraProvider>
      );
    });
    await settle();
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
  // The select's menu renders after a frame; the provider list after its fetch settles.
  const settle = () =>
    act(
      () =>
        new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 50);
        })
    );
  // The provider is the section's only select; inside its Field it takes the field's label.
  const providerTrigger = () => host.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]');

  beforeEach(() => {
    api.getExternalProviderConfigs.mockClear();
    api.getStarterModels.mockClear();
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

  it('refuses to save an invalid record, marking the mode group and moving focus to it', async () => {
    await render(gptImage);

    await click(checkbox('models.externalModes.txt2img'));
    await click(checkbox('models.externalModes.img2img'));
    await click(button('models.saveExternalSettings'));

    const group = host.querySelector<HTMLElement>('[role="group"][aria-labelledby]')!;
    const alert = group.querySelector('[role="alert"]')!;
    expect(document.getElementById(group.getAttribute('aria-labelledby')!)?.textContent).toBe(
      'models.externalFields.supportedModes'
    );
    expect(alert.textContent).toBe('Choose at least one supported mode.');
    expect(group.getAttribute('aria-describedby')).toBe(alert.id);
    expect(host.querySelectorAll('[role="alert"]')).toHaveLength(1);
    expect(checkbox('models.externalModes.inpaint').getAttribute('aria-invalid')).toBe('true');
    expect(document.activeElement).toBe(checkbox('models.externalModes.txt2img'));
    expect(api.updateModel).not.toHaveBeenCalled();
    // Reset discards the draft and the error together.
    await click(button('common.reset'));
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(checkbox('models.externalModes.inpaint').getAttribute('aria-invalid')).not.toBe('true');
    expect(checkbox('models.externalModes.txt2img').checked).toBe(true);
  });

  it('marks the limit field a failed validation names and describes it with the message', async () => {
    await render(gptImage);

    await type(input('models.externalFields.maxImageWidth'), '2048');
    // Focus and blur in one task leave the number input a refocus for the next frame; a user's click comes later.
    await settle();
    await click(button('models.saveExternalSettings'));

    const height = input('models.externalFields.maxImageHeight');
    const message = [...host.querySelectorAll('[role="alert"]')];
    expect(message.map((element) => element.textContent)).toEqual([
      'Set both a max width and a max height, or neither.',
    ]);
    expect(height.getAttribute('aria-invalid')).toBe('true');
    expect(height.getAttribute('aria-describedby')?.split(' ')).toContain(message[0]!.id);
    expect(input('models.externalFields.maxImageWidth').getAttribute('aria-invalid')).toBeNull();
    expect(document.activeElement).toBe(height);
    expect(api.updateModel).not.toHaveBeenCalled();

    // Editing clears the marking along with the message.
    await type(height, '2048');
    expect(height.getAttribute('aria-invalid')).toBeNull();
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('keeps the saved unregistered provider selectable after switching without discarding other edits', async () => {
    await render({ ...gptImage, provider_id: 'custom' });
    await settle();

    expect(host.querySelector('input[aria-label="models.providerId"]')).toBeNull();
    expect(providerTrigger()?.textContent).toContain('custom');
    await click(providerTrigger()!);
    await settle();

    const options = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    expect(options.map((option) => option.textContent)).toEqual(['openai', 'gemini', 'custom']);
    await click(options[1]!);
    await settle();

    expect(providerTrigger()?.textContent).toContain('gemini');
    await type(input('models.externalFields.maxReferenceImages'), '4');
    await click(providerTrigger()!);
    await settle();
    const savedProvider = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(
      (option) => option.textContent === 'custom'
    );
    expect(savedProvider).toBeDefined();
    await click(savedProvider!);
    await settle();
    expect(providerTrigger()?.textContent).toContain('custom');
    expect(getModelDraft('gpt-image-1')?.external?.fields).toEqual({ maxReferenceImages: 4 });
  });

  it('keeps the provider a text field while the provider list is unavailable', async () => {
    api.getExternalProviderConfigs.mockRejectedValueOnce(new Error('offline'));
    await render(gptImage);
    await settle();

    const providerInput = host.querySelector<HTMLInputElement>('input[aria-label="models.providerId"]')!;
    expect(providerInput.value).toBe('openai');
    await type(providerInput, 'seedream');
    expect(getModelDraft('gpt-image-1')?.external?.fields).toEqual({ providerId: 'seedream' });
  });

  it('locks routing and capability edits for a starter-managed record', async () => {
    const geminiFlash: ExternalModelSettingsModel = {
      ...gptImage,
      capabilities: { max_images_per_request: 1, modes: ['txt2img'], supports_reference_images: true },
      key: 'gemini-flash',
      provider_id: 'gemini',
      provider_model_id: 'gemini-2.5-flash-image',
      source: 'external://gemini/gemini-2.5-flash-image',
    };
    api.updateModel.mockImplementation((_key: string, changes: Record<string, unknown>) =>
      Promise.resolve({ ...geminiFlash, ...changes })
    );
    await render(geminiFlash);

    expect(host.querySelector('[role="note"]')?.textContent).toBe('models.externalFields.starterManaged');
    expect(host.querySelectorAll('input[type="checkbox"]')).toHaveLength(0);
    expect(host.querySelector('input[aria-label="models.externalFields.maxImagesPerRequest"]')).toBeNull();
    expect(host.textContent).toContain('models.externalModes.txt2img');
    expect(host.textContent).toContain('1024');

    expect(providerTrigger()?.disabled).toBe(true);
    expect(host.querySelector<HTMLInputElement>('input[value="gemini-2.5-flash-image"]')?.disabled).toBe(true);
    expect(button('models.saveExternalSettings').disabled).toBe(true);
    await click(button('models.saveExternalSettings'));
    expect(api.updateModel).not.toHaveBeenCalled();
  });

  it('blocks edits while catalog discovery loads or fails and enables custom records after Retry', async () => {
    let rejectCatalog!: (error: Error) => void;
    api.getStarterModels.mockImplementationOnce(
      () =>
        new Promise((_resolve, reject) => {
          rejectCatalog = reject;
        })
    );
    await render(gptImage);
    expect(host.querySelector('[role="status"]')?.textContent).toContain('models.externalFields.catalogLoading');
    expect(providerTrigger()?.disabled).toBe(true);
    expect(host.querySelector<HTMLInputElement>('input[value="gpt-image-1"]')?.disabled).toBe(true);
    expect(host.querySelectorAll('input[type="checkbox"]')).toHaveLength(0);

    await act(() => rejectCatalog(new Error('offline')));
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('models.externalFields.catalogUnavailable');
    expect(button('models.saveExternalSettings').disabled).toBe(true);
    expect(host.querySelectorAll('input[type="checkbox"]')).toHaveLength(0);
    expect(getModelDraft('gpt-image-1')?.external).toBeUndefined();

    let resolveCatalog!: (response: Awaited<ReturnType<typeof api.getStarterModels>>) => void;
    api.getStarterModels.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveCatalog = resolve;
        })
    );
    const retry = button('common.retry');
    retry.focus();
    await click(retry);
    // The button stays mounted and focused while the catalog reloads, and refuses a second request.
    expect(document.activeElement).toBe(retry);
    expect(retry.getAttribute('aria-busy')).toBe('true');
    expect(retry.getAttribute('aria-disabled')).toBe('true');
    await click(retry);
    expect(api.getStarterModels).toHaveBeenCalledTimes(2);

    await act(async () => {
      resolveCatalog({ starter_bundles: {}, starter_models: [] });
      await Promise.resolve();
    });
    await settle();
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(providerTrigger()?.disabled).toBe(false);
    // Focus moves to the editor the catalog unlocked rather than falling to the page.
    expect(document.activeElement).toBe(providerTrigger());
    expect(checkbox('models.externalModes.txt2img').checked).toBe(true);
    expect(getModelDraft('gpt-image-1')?.external).toBeUndefined();
  });

  it('moves focus to the catalog note when a retry finds the record catalog-managed', async () => {
    api.getStarterModels.mockRejectedValueOnce(new Error('offline'));
    api.getStarterModels.mockResolvedValueOnce({ starter_bundles: {}, starter_models: [{ source: gptImage.source! }] });
    await render(gptImage);

    const retry = button('common.retry');
    retry.focus();
    await click(retry);
    await settle();

    expect(document.activeElement?.textContent).toBe('models.externalFields.starterManaged');
  });

  it('explains a retained draft while a failed catalog fetch keeps the record locked', async () => {
    api.getStarterModels.mockRejectedValueOnce(new Error('offline'));
    const saved = toExternalModelValues(gptImage);
    recordModelDraftFields(gptImage.key, 'external', { ...saved, providerModelId: 'draft-model' }, saved);
    await render(gptImage);

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('models.externalFields.catalogUnavailable');
    expect(host.querySelector('[role="note"]')?.textContent).toBe('models.externalFields.lockedDraftBlocked');
    expect(host.querySelector<HTMLInputElement>('input[value="draft-model"]')?.disabled).toBe(true);
    expect(button('models.saveExternalSettings').disabled).toBe(true);
    expect(button('common.reset')).toBeDefined();
  });

  it('keeps a retained draft visible but unsaveable when retry discovers a catalog-managed record', async () => {
    api.getStarterModels.mockRejectedValueOnce(new Error('offline'));
    api.getStarterModels.mockResolvedValueOnce({
      starter_bundles: {},
      starter_models: [{ source: gptImage.source! }],
    });
    const saved = toExternalModelValues(gptImage);
    recordModelDraftFields(
      gptImage.key,
      'external',
      {
        ...saved,
        providerModelId: 'draft-model',
        maxImagesPerRequest: 7,
      },
      saved
    );
    await render(gptImage);
    expect(button('models.saveExternalSettings').disabled).toBe(true);
    await click(button('common.retry'));
    await settle();

    expect(host.textContent).toContain('models.externalFields.lockedDraftBlocked');
    expect(host.querySelector<HTMLInputElement>('input[value="draft-model"]')?.disabled).toBe(true);
    expect(host.textContent).toContain('7');
    expect(button('models.saveExternalSettings').disabled).toBe(true);
    await click(button('models.saveExternalSettings'));
    expect(api.updateModel).not.toHaveBeenCalled();
    await click(button('common.reset'));
    expect(getModelDraft(gptImage.key)?.external).toBeUndefined();
    expect(host.textContent).not.toContain('models.externalFields.lockedDraftBlocked');
    expect(host.querySelector<HTMLInputElement>('input[value="gpt-image-1"]')?.disabled).toBe(true);
    expect(host.textContent).toContain('10');
  });

  it('surfaces a refused save as the readable validation message and keeps the edits for another attempt', async () => {
    api.updateModel.mockRejectedValue(
      new ApiError(
        JSON.stringify({
          detail: [
            {
              ctx: { gt: 0 },
              input: 0,
              loc: ['body', 'changes', 'capabilities', 'max_images_per_request'],
              msg: 'Input should be greater than 0',
              type: 'greater_than',
            },
          ],
        }),
        422
      )
    );
    await render(gptImage);

    await click(checkbox('models.externalFields.supportsSeed'));
    await click(button('models.saveExternalSettings'));

    expect(onError).toHaveBeenCalledOnce();
    const [message] = onError.mock.calls[0]!;
    expect(message).toContain('max images per request');
    expect(message).not.toContain('{');
    expect(checkbox('models.externalFields.supportsSeed').checked).toBe(true);
    expect(button('models.saveExternalSettings').disabled).toBe(false);
  });

  it('shows what the backend saved, not what was sent, once the save lands', async () => {
    setModelsSnapshotForTests({ models: [gptImage as ModelConfig], status: 'loaded' });
    // The backend normalizes the route it stores.
    api.updateModel.mockImplementation((_key: string, changes: Record<string, unknown>) =>
      Promise.resolve({ ...gptImage, ...changes, provider_model_id: 'gpt-image-1.5-2026' })
    );
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <LibraryBackedSection onError={onError} onSaved={onSaved} />
        </ChakraProvider>
      );
    });
    await settle();

    await type(host.querySelector<HTMLInputElement>('input[value="gpt-image-1"]')!, 'gpt-image-1.5');
    await click(button('models.saveExternalSettings'));

    expect(onSaved).toHaveBeenCalledOnce();
    expect(host.querySelector('input[value="gpt-image-1.5-2026"]')).not.toBeNull();
    expect(host.textContent).not.toContain('models.unsavedChanges');
    expect(button('models.saveExternalSettings').disabled).toBe(true);
  });
});
