import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { type ModelsUiAdapter, ModelsUiProvider, NO_DEFAULTS_RULES } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelDetail } from './ModelDetail';

const { section } = vi.hoisted(() => ({
  section: (name: string) => () => <section data-section={name} />,
}));

vi.mock('./DefaultSettingsSection', () => ({ DefaultSettingsSection: section('default-settings') }));
vi.mock('./RelatedModelsSection', () => ({ RelatedModelsSection: section('related-models') }));
vi.mock('./TriggerPhrasesEditor', () => ({ MemoizedTriggerPhrasesEditor: section('trigger-phrases') }));
vi.mock('./WorkingMemorySection', () => ({ WorkingMemorySection: section('working-memory') }));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t: (key: string) => key }),
}));

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
  defaultsRules: NO_DEFAULTS_RULES,
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

  it('shows preserved source links once alongside the original download source', async () => {
    const source = 'https://example.com/model';
    setModelsSnapshotForTests({
      models: [{ ...MAIN, source_url: source, source_urls: [source, 'https://example.org/model'] }],
      status: 'loaded',
    });
    await renderDetail(MAIN.key);

    const links = [...host.querySelectorAll<HTMLAnchorElement>('a[target="_blank"]')];
    expect(links.map((link) => link.href)).toEqual([source, 'https://example.org/model']);
    expect(links.map((link) => link.closest('dd')?.previousElementSibling?.textContent)).toEqual([
      'models.sourceUrl',
      'models.modelPage',
    ]);
    expect(host.textContent).toContain(MAIN.source);
  });

  it('does not repeat the Hugging Face page the source already links to', async () => {
    const page = 'https://huggingface.co/org/repo';
    setModelsSnapshotForTests({
      models: [{ ...MAIN, source: 'org/repo::model.safetensors', source_type: 'hf_repo_id', source_urls: [page] }],
      status: 'loaded',
    });
    await renderDetail(MAIN.key);

    expect([...host.querySelectorAll<HTMLAnchorElement>('a[target="_blank"]')].map((link) => link.href)).toEqual([
      page,
    ]);
  });

  it('shows unsafe source links as text instead of anchors', async () => {
    setModelsSnapshotForTests({
      models: [
        { ...MAIN, source_url: 'javascript:alert(1)', source_urls: ['javascript:alert(2)', 'httpx://example.com'] },
      ],
      status: 'loaded',
    });
    await renderDetail(MAIN.key);

    expect(host.querySelector('a[target="_blank"]')).toBeNull();
    expect(host.textContent).toContain('javascript:alert(1)');
    expect(host.textContent).toContain('javascript:alert(2)');
    expect(host.textContent).toContain('httpx://example.com');
  });

  it('lists the roles a model also serves, and no row without any', async () => {
    const withRoles = config('llm', 'text_llm', {
      roles: [
        { type: 'qwen3_5_encoder', variant: 'qwen3_5_9b' },
        { format: null, type: 'qwen3_encoder' },
      ],
    });
    setModelsSnapshotForTests({ models: [MAIN, withRoles], status: 'loaded' });
    const alsoUsableAs = () =>
      [...host.querySelectorAll('dt')].find((label) => label.textContent === 'models.alsoUsableAs')?.nextElementSibling
        ?.textContent;

    await renderDetail(withRoles.key);
    expect(alsoUsableAs()).toBe('Qwen3.5 Encoder · Qwen3.5 9B (MOSS-Image) and Qwen3 Encoder');

    await renderDetail(MAIN.key);
    expect(alsoUsableAs()).toBeUndefined();
  });

  it('offers Related Models to a Text LLM only when it serves as a linkable type', async () => {
    const encoderLlm = config('encoder-llm', 'text_llm', {
      roles: [{ type: 'qwen3_5_encoder', variant: 'qwen3_5_9b' }],
    });
    const plainLlm = config('plain-llm', 'text_llm', { roles: [] });
    setModelsSnapshotForTests({ models: [encoderLlm, plainLlm], status: 'loaded' });
    const hasRelated = () => host.querySelector('[data-section="related-models"]') !== null;

    await renderDetail(encoderLlm.key);
    expect(hasRelated()).toBe(true);

    await renderDetail(plainLlm.key);
    expect(hasRelated()).toBe(false);
  });

  it('offers Run on CPU exactly when the config carries cpu_only', async () => {
    setModelsSnapshotForTests({ models: [GEMMA4_ENCODER, QWEN3_VL_ENCODER_WITHOUT_CPU_ONLY], status: 'loaded' });
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

    await renderDetail(CONTROL_LORA.key);
    expect(hasTriggerPhrases()).toBe(false);
  });
});
