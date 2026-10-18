import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
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

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useCapabilities: () => ({ canManageModels: true }),
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

const ignoreConfirm = () => undefined;

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

  it('places working memory last, after the trigger phrases', async () => {
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <ModelDetail modelKey={MAIN.key} onRequestConfirm={ignoreConfirm} />
        </ChakraProvider>
      )
    );

    const order = [...host.querySelectorAll('[data-section]')].map((element) => element.getAttribute('data-section'));
    expect(order).toEqual(['default-settings', 'related-models', 'trigger-phrases', 'working-memory']);
  });
});
