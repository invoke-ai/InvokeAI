import type { ModelConfig } from '@features/models';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ModelSelect } from './ModelSelect';

// Browser tests run without an i18n provider, so translated copy renders as
// the raw key (e.g. 'models.compactRows'); assertions below match the keys.
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

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
const loraModel = {
  ...model,
  base: 'sd-1',
  key: 'sd1-lora',
  name: 'Detail LoRA',
  type: 'lora',
} as ModelConfig;
const secondSdxlModel = { ...model, key: 'sdxl-other', name: 'Another SDXL' } as ModelConfig;
const MODELS_UI_ADAPTER = { enableModelDescriptions: true, managerProjectId: null };
const MAIN_MODEL_TYPES: ['main'] = ['main'];
const CROSS_TYPE_MODEL_TYPES: ['main', 'lora'] = ['main', 'lora'];

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

  const renderPicker = async (props: Partial<Parameters<typeof ModelSelect>[0]> = {}) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect
              modelTypes={MAIN_MODEL_TYPES}
              showManagerButton={false}
              value={null}
              onChange={vi.fn()}
              {...props}
            />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);
  };

  const optionTexts = (): string[] =>
    [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? '');

  // The bug that prompted the rewrite: grouping by (type, base) printed "Main
  // Models" above every base section of a cross-type picker.
  it('groups a cross-type picker by base alone, with no type header', async () => {
    setModelsSnapshotForTests({ error: null, models: [model, loraModel], status: 'loaded' });
    await renderPicker({ id: 'test-cross-type', modelTypes: CROSS_TYPE_MODEL_TYPES });

    const listbox = document.querySelector('[role="listbox"]')!;

    expect(listbox.textContent).toContain('Stable Diffusion 1.x');
    expect(listbox.textContent).toContain('Stable Diffusion XL');
    expect(listbox.textContent).not.toContain('Main Models');
    // Type is on the row instead, so the two rows stay distinguishable.
    expect(optionTexts().join(' ')).toContain('LoRA');
  });

  it('moves the active row across a group boundary and selects with Enter', async () => {
    const onChange = vi.fn();

    setModelsSnapshotForTests({ error: null, models: [model, loraModel], status: 'loaded' });
    await renderPicker({ id: 'test-keyboard', modelTypes: CROSS_TYPE_MODEL_TYPES, onChange });

    const search = document
      .querySelector<HTMLInputElement>('[role="listbox"]')!
      .closest('[data-scope="popover"]')!
      .querySelector<HTMLInputElement>('input')!;

    // sd-1 sorts before sdxl, so the LoRA leads and one step lands on the main
    // model in the next group.
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('Detail LoRA');

    await act(() => {
      search.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'ArrowDown' }));
    });
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('SDXL Main');

    await act(() => {
      search.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter' }));
    });

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ key: 'sdxl-main' }));
  });

  it('remembers the compact row density across a close and reopen', async () => {
    setModelsSnapshotForTests({ error: null, models: [model, secondSdxlModel], status: 'loaded' });
    await renderPicker({ id: 'test-compact' });

    const toggle = document.querySelector<HTMLButtonElement>('button[aria-label="models.compactRows"]')!;

    expect(toggle).not.toBeNull();
    await act(() => toggle.click());
    await expect.poll(() => document.querySelector('button[aria-label="models.fullRows"]')).not.toBeNull();

    // Close and reopen: the preference is stored per picker id, not per mount.
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());

    await expect.poll(() => document.querySelector('button[aria-label="models.fullRows"]')).not.toBeNull();
  });
});
