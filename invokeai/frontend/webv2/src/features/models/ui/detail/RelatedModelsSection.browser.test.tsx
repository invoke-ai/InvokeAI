import type { ModelConfig } from '@features/models';
import type { ArchitectureCapabilitiesRow } from '@features/models/architectures';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/models/core/architectureCapabilities.testing';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { setRelationshipsSnapshotForTests } from '@features/models/data/relationshipsStore';
import { ModelsUiProvider, NO_SCHEDULER_POLICY } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { RelatedModelsSection } from './RelatedModelsSection';

const api = vi.hoisted(() => ({
  addModelRelationship: vi.fn(),
  getRelatedModelKeys: vi.fn(),
  removeModelRelationship: vi.fn(),
}));

vi.mock('@features/models/data/relationshipsApi', () => api);

const getArchitectureCapabilities = vi.hoisted(() => vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>());
vi.mock('@features/models/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => getArchitectureCapabilities(),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const createModel = (overrides: Partial<ModelConfig>): ModelConfig =>
  ({
    base: 'sdxl',
    description: null,
    file_size: 1024,
    format: 'checkpoint',
    hash: 'hash',
    key: 'key',
    name: 'Model',
    source: 'model.safetensors',
    type: 'main',
    ...overrides,
  }) as ModelConfig;

const fluxMain = createModel({ base: 'flux', key: 'flux-main', name: 'FLUX Dev', type: 'main' });
const fluxMainSibling = createModel({ base: 'flux', key: 'flux-main-2', name: 'FLUX Krea', type: 'main' });
const fluxLora = createModel({ base: 'flux', key: 'flux-lora', name: 'FLUX LoRA', type: 'lora' });
const sdxlLora = createModel({ base: 'sdxl', key: 'sdxl-lora', name: 'SDXL LoRA', type: 'lora' });
const t5Encoder = createModel({ base: 'any', key: 't5-xxl', name: 'T5 XXL', type: 't5_encoder' });
const externalMain = createModel({ base: 'external', key: 'gpt-image', name: 'GPT Image', type: 'main' });
const fluxVae = createModel({ base: 'flux', key: 'flux-vae', name: 'FLUX VAE', type: 'vae' });
const sdxlVae = createModel({ base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' });

const MODELS_UI_ADAPTER = {
  canManageModels: true,
  enableModelDescriptions: false,
  installInPlace: true,
  isProjectActive: () => true,
  managerProjectId: null,
  schedulerPolicy: NO_SCHEDULER_POLICY,
  setInstallInPlace: () => undefined,
};
const SECTION_MODEL = { base: 'flux', key: 'flux-main', type: 'main' } as const;

const deferredTable = () => {
  let resolve: (rows: ArchitectureCapabilitiesRow[]) => void = () => undefined;
  let reject: (error: Error) => void = () => undefined;
  const promise = new Promise<ArchitectureCapabilitiesRow[]>((settle, fail) => {
    resolve = settle;
    reject = fail;
  });

  return { promise, reject, resolve };
};

describe('RelatedModelsSection', () => {
  let host: HTMLDivElement;
  let root: Root;
  // Each test answers the table request itself. Once any test has asked for the table, the store asks again as the
  // account activates, so the request may predate the render.
  let table: ReturnType<typeof deferredTable>;

  beforeEach(() => {
    api.addModelRelationship.mockReset().mockResolvedValue(undefined);
    api.getRelatedModelKeys.mockReset().mockResolvedValue([]);
    api.removeModelRelationship.mockReset().mockResolvedValue(undefined);
    table = deferredTable();
    getArchitectureCapabilities.mockReset().mockImplementation(() => table.promise);
    accountLifecycle.activate('related-models', ':user:related-models');
    setModelsSnapshotForTests({
      error: null,
      models: [fluxMain, fluxMainSibling, fluxLora, sdxlLora, t5Encoder, externalMain, fluxVae, sdxlVae],
      status: 'loaded',
    });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
    setModelsSnapshotForTests({ error: null, models: [], status: 'idle' });
    setRelationshipsSnapshotForTests({ relatedKeysByModelKey: {} });
    // Returns the architecture table's store and registry to their unloaded state.
    accountLifecycle.invalidate();
  });

  const renderSection = async (
    model: Parameters<typeof RelatedModelsSection>[0]['model'] = SECTION_MODEL,
    onError: (message: string) => void = vi.fn()
  ) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <RelatedModelsSection model={model} onError={onError} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });
  };

  const trigger = () => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]');
  const openPicker = async () => {
    await act(() => trigger()?.click());
  };

  const renderSectionAndOpenPicker = async (onError: (message: string) => void = vi.fn()) => {
    await renderSection(SECTION_MODEL, onError);
    await openPicker();
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);
  };

  const optionTexts = (): string[] =>
    [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? '');
  const pickerStatus = (): string =>
    document.querySelector('[data-scope="popover"][data-part="content"] [role="status"]')?.textContent ?? '';

  it('offers only base-compatible candidates for a flux main', async () => {
    await renderSectionAndOpenPicker();

    const options = optionTexts().join(' ');

    expect(options).toContain('FLUX LoRA');
    // T5 has base `any` but a curated allowance for flux.
    expect(options).toContain('T5 XXL');
    expect(options).not.toContain('SDXL LoRA');
    expect(options).not.toContain('GPT Image');
    // The subject model never offers itself.
    expect(options).not.toContain('FLUX Dev');
    // Single-slot types never offer their own type: no second main.
    expect(options).not.toContain('FLUX Krea');
  });

  it('links a model and renders it without refetching', async () => {
    await renderSectionAndOpenPicker();

    const t5Option = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
      option.textContent?.includes('T5 XXL')
    );

    expect(t5Option).toBeDefined();
    await act(() => t5Option?.click());

    expect(api.addModelRelationship).toHaveBeenCalledWith('flux-main', 't5-xxl', expect.anything());
    // The shared store patch renders the row; no second fetch happens.
    await expect.poll(() => host.textContent).toContain('T5 XXL');
    expect(api.getRelatedModelKeys).toHaveBeenCalledTimes(1);
  });

  it('unwraps a backend detail body into the error callback', async () => {
    const onError = vi.fn();

    api.addModelRelationship.mockRejectedValue(new ApiError('{"detail":"Models are incompatible"}', 409));
    await renderSectionAndOpenPicker(onError);

    const t5Option = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
      option.textContent?.includes('T5 XXL')
    );

    await act(() => t5Option?.click());

    await expect.poll(() => onError.mock.calls.length).toBeGreaterThan(0);
    expect(onError).toHaveBeenCalledWith('Models are incompatible');
  });

  it('shows persisted links but no picker for an unlinkable base', async () => {
    api.getRelatedModelKeys.mockResolvedValue(['flux-main']);
    await renderSection({ base: 'external', key: 'gpt-image', type: 'main' });

    // The wildcard-era link renders with its unlink affordance...
    await expect.poll(() => host.textContent).toContain('FLUX Dev');
    expect(host.querySelector('[aria-label*="unlinkNamed"]')).not.toBeNull();
    // ...but an external-base subject can never match, so no add-picker.
    expect(host.querySelector('[aria-haspopup="listbox"]')).toBeNull();
  });

  const answerTable = async (answer: () => void) => {
    await act(async () => {
      answer();
      await table.promise.catch(() => undefined);
    });
  };

  it('offers VAEs only once the architecture table says which ones fit', async () => {
    await renderSectionAndOpenPicker();

    expect(getArchitectureCapabilities).toHaveBeenCalledTimes(1);
    expect(optionTexts().join(' ')).toContain('FLUX LoRA');
    expect(optionTexts().join(' ')).not.toContain('VAE');
    expect(pickerStatus()).toContain('models.vaePairingLoading');

    await answerTable(() => table.resolve(architectureCapabilitiesFixture));

    await expect.poll(() => optionTexts().join(' ')).toContain('FLUX VAE');
    expect(optionTexts().join(' ')).not.toContain('SDXL VAE');
    expect(pickerStatus()).not.toContain('models.vaePairing');
  });

  it('says VAE links are missing after the table fails, and retries', async () => {
    await renderSectionAndOpenPicker();
    await answerTable(() => table.reject(new Error('outage')));

    await expect.poll(pickerStatus).toContain('models.vaePairingLoadFailed');
    expect(optionTexts().join(' ')).toContain('FLUX LoRA');
    expect(optionTexts().join(' ')).not.toContain('FLUX VAE');

    const retry = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'common.retry'
    );

    expect(retry).toBeDefined();
    getArchitectureCapabilities.mockResolvedValue(architectureCapabilitiesFixture);
    await act(() => retry?.click());

    await expect.poll(() => optionTexts().join(' ')).toContain('FLUX VAE');
    expect(pickerStatus()).not.toContain('models.vaePairing');
    expect(getArchitectureCapabilities).toHaveBeenCalledTimes(2);
  });

  it("keeps a VAE's picker usable while the table decides every candidate", async () => {
    // An SDXL VAE's every installed candidate is a VAE pair: nothing is decided until the table loads.
    await renderSection({ base: 'sdxl', key: 'sdxl-vae', type: 'vae' });

    expect(trigger()?.disabled).toBe(false);
    await openPicker();
    await expect.poll(pickerStatus).toContain('models.vaePairingLoading');
    expect(optionTexts()).toEqual([]);

    await answerTable(() => table.resolve(architectureCapabilitiesFixture));

    await expect.poll(() => optionTexts().join(' ')).toContain('SDXL LoRA');
    expect(optionTexts().join(' ')).not.toContain('FLUX');
  });

  it('stays open, with focus inside, when the table admits none of the candidates', async () => {
    const cogViewVae = createModel({ base: 'cogview4', key: 'cogview4-vae', name: 'CogView4 VAE', type: 'vae' });

    setModelsSnapshotForTests({
      error: null,
      models: [fluxMain, fluxLora, sdxlLora, t5Encoder, cogViewVae],
      status: 'loaded',
    });
    await renderSection({ base: 'cogview4', key: 'cogview4-vae', type: 'vae' });
    await openPicker();
    await expect.poll(pickerStatus).toContain('models.vaePairingLoading');

    const popover = () => document.querySelector('[data-scope="popover"][data-part="content"]');
    const search = popover()?.querySelector<HTMLInputElement>('input[role="combobox"]');
    search?.focus();

    await answerTable(() => table.resolve(architectureCapabilitiesFixture));

    await expect.poll(pickerStatus).toContain('models.scopeNoCompatibleInstalled');
    // Still open, not merely fading out: the trigger reports it, and focus stays in the popup.
    expect(trigger()?.getAttribute('aria-expanded')).toBe('true');
    expect(popover()?.getAttribute('data-state')).toBe('open');
    expect(popover()?.contains(document.activeElement)).toBe(true);
  });
});
