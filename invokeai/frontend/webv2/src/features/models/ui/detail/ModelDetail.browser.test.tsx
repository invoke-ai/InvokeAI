import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { type ModelsUiAdapter, ModelsUiProvider, NO_SCHEDULER_POLICY } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelDetail } from './ModelDetail';

// The sections own their data; here they only mark where the page places them.
const { section } = vi.hoisted(() => ({
  section: (name: string) => () => <section data-section={name} />,
}));

vi.mock('./DefaultSettingsSection', () => ({ DefaultSettingsSection: section('default-settings') }));
vi.mock('./RelatedModelsSection', () => ({ RelatedModelsSection: section('related-models') }));
vi.mock('./TriggerPhrasesEditor', () => ({ MemoizedTriggerPhrasesEditor: section('trigger-phrases') }));
vi.mock('./WorkingMemorySection', () => ({ WorkingMemorySection: section('working-memory') }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const MAIN = {
  base: 'sdxl',
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash-main',
  key: 'main-1',
  name: 'Main Model',
  path: '/models/main.safetensors',
  source: '/models/main.safetensors',
  source_type: 'path',
  trigger_phrases: [],
  type: 'main',
  variant: 'normal',
} as unknown as ModelConfig;

// The backend serializes `cpu_only` and `trigger_phrases` only on configs that store them, whatever the type.
const config = (key: string, type: string, fields: Partial<ModelConfig>) =>
  ({
    base: 'any',
    file_size: 1024,
    format: 'checkpoint',
    hash: `hash-${key}`,
    key,
    name: key,
    path: `/models/${key}.safetensors`,
    source: `/models/${key}.safetensors`,
    source_type: 'path',
    type,
    ...fields,
  }) as unknown as ModelConfig;

const GEMMA4_ENCODER = config('gemma4-encoder', 'gemma4_encoder', { cpu_only: null });
const QWEN3_VL_ENCODER_WITHOUT_CPU_ONLY = config('qwen3-vl-encoder', 'qwen3_vl_encoder', {});
const LORA = config('lora-1', 'lora', { trigger_phrases: null });
const EMBEDDING = config('embedding-1', 'embedding', {});
const CONTROL_LORA = config('control-lora-1', 'control_lora', { trigger_phrases: null });

const ignoreConfirm = () => undefined;

const MANAGER: ModelsUiAdapter = {
  canManageModels: true,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => false,
  managerProjectId: null,
  schedulerPolicy: NO_SCHEDULER_POLICY,
  setInstallInPlace: () => undefined,
};

describe('ModelDetail', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    accountLifecycle.activate('model-detail-test', ':user:model-detail-test');
    setModelsSnapshotForTests({ models: [MAIN], status: 'loaded' });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setModelsSnapshotForTests({ models: [], status: 'idle' });
    accountLifecycle.invalidate();
  });

  const renderDetail = (modelKey: string) =>
    act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MANAGER}>
            <ModelDetail key={modelKey} modelKey={modelKey} onRequestConfirm={ignoreConfirm} />
          </ModelsUiProvider>
        </ChakraProvider>
      )
    );

  it('places working memory last, after the trigger phrases', async () => {
    await renderDetail(MAIN.key);

    const order = [...host.querySelectorAll('[data-section]')].map((element) => element.getAttribute('data-section'));
    expect(order).toEqual(['default-settings', 'related-models', 'trigger-phrases', 'working-memory']);
  });

  it('offers Run on CPU exactly when the config carries cpu_only', async () => {
    setModelsSnapshotForTests({ models: [GEMMA4_ENCODER, QWEN3_VL_ENCODER_WITHOUT_CPU_ONLY], status: 'loaded' });
    // A checkbox named by its label, not merely the label text somewhere on the page.
    const hasCpuSwitch = () =>
      [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].some(
        (input) => input.labels?.[0]?.textContent === 'models.runOnCpu'
      );

    await renderDetail(GEMMA4_ENCODER.key);
    expect(hasCpuSwitch()).toBe(true);

    await renderDetail(QWEN3_VL_ENCODER_WITHOUT_CPU_ONLY.key);
    expect(hasCpuSwitch()).toBe(false);
  });

  it('offers trigger phrases when the config carries trigger_phrases, except on Control LoRAs', async () => {
    setModelsSnapshotForTests({ models: [LORA, EMBEDDING, CONTROL_LORA], status: 'loaded' });
    const hasTriggerPhrases = () => host.querySelector('[data-section="trigger-phrases"]') !== null;

    await renderDetail(LORA.key);
    expect(hasTriggerPhrases()).toBe(true);

    await renderDetail(EMBEDDING.key);
    expect(hasTriggerPhrases()).toBe(false);

    // Prompt triggers never read Control LoRAs, so phrases stored there would have no effect.
    await renderDetail(CONTROL_LORA.key);
    expect(hasTriggerPhrases()).toBe(false);
  });
});
