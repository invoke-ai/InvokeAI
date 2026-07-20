import type { ModelConfig } from '@features/models';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelSelect } from './ModelSelect';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = {
  base: 'sdxl',
  description: null,
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash',
  key: 'sdxl-main',
  name: 'SDXL Main',
  source: 'sdxl.safetensors',
  type: 'main',
} as ModelConfig;
const MODELS_UI_ADAPTER = { enableModelDescriptions: true, managerProjectId: null };
const MAIN_MODEL_TYPES: ['main'] = ['main'];

describe('ModelSelect loading states', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
    setModelsSnapshotForTests({ error: null, models: [], status: 'idle' });
  });

  it('shows compatible installed models through the public picker', async () => {
    setModelsSnapshotForTests({ error: null, models: [model], status: 'loaded' });

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect modelTypes={MAIN_MODEL_TYPES} showManagerButton={false} value={null} onChange={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());

    await expect.poll(() => document.querySelector('[role="option"]')?.textContent).toContain('SDXL Main');
  });
});
