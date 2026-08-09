import type { ModelConfig } from '@features/models';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { setRelationshipsSnapshotForTests } from '@features/models/data/relationshipsStore';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
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

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...((await importOriginal()) as Record<string, unknown>),
  ...api,
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
const fluxLora = createModel({ base: 'flux', key: 'flux-lora', name: 'FLUX LoRA', type: 'lora' });
const sdxlLora = createModel({ base: 'sdxl', key: 'sdxl-lora', name: 'SDXL LoRA', type: 'lora' });
const t5Encoder = createModel({ base: 'any', key: 't5-xxl', name: 'T5 XXL', type: 't5_encoder' });
const externalMain = createModel({ base: 'external', key: 'gpt-image', name: 'GPT Image', type: 'main' });

const MODELS_UI_ADAPTER = { enableModelDescriptions: false, managerProjectId: null };
const SECTION_MODEL = { base: 'flux', key: 'flux-main', type: 'main' } as const;

describe('RelatedModelsSection', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    api.addModelRelationship.mockReset().mockResolvedValue(undefined);
    api.getRelatedModelKeys.mockReset().mockResolvedValue([]);
    api.removeModelRelationship.mockReset().mockResolvedValue(undefined);
    accountLifecycle.activate('related-models', ':user:related-models');
    setModelsSnapshotForTests({
      error: null,
      models: [fluxMain, fluxLora, sdxlLora, t5Encoder, externalMain],
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
  });

  const renderSection = async (model: Parameters<typeof RelatedModelsSection>[0]['model'] = SECTION_MODEL) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <RelatedModelsSection model={model} onError={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });
  };

  const renderSectionAndOpenPicker = async () => {
    await renderSection();
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);
  };

  const optionTexts = (): string[] =>
    [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? '');

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

  it('shows persisted links but no picker for an unlinkable base', async () => {
    api.getRelatedModelKeys.mockResolvedValue(['flux-main']);
    await renderSection({ base: 'external', key: 'gpt-image', type: 'main' });

    // The wildcard-era link renders with its unlink affordance...
    await expect.poll(() => host.textContent).toContain('FLUX Dev');
    expect(host.querySelector('[aria-label*="unlinkNamed"]')).not.toBeNull();
    // ...but an external-base subject can never match, so no add-picker.
    expect(host.querySelector('[aria-haspopup="listbox"]')).toBeNull();
  });
});
