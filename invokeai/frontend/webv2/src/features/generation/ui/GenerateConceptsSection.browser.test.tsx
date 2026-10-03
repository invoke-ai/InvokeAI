import type {
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
} from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { flushGenerateDrafts } from '@features/generation/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { GenerateSettingsUpdate } from './generateDebounce';

import { GenerateConceptsContent } from './GenerateConceptsSection';
import { GenerationUiProvider, type GenerationUiAdapter } from './GenerationUiContext';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const LORA_MODEL: LoraModelConfig = { base: 'sdxl', key: 'lora-1', name: 'Ink Wash', type: 'lora' };
const MAIN_MODEL = { base: 'sdxl', key: 'main-1', name: 'Base', type: 'main' } as GenerateModelConfig;
const LORA_MODELS = [LORA_MODEL];
const LORA: GenerateLora = { isEnabled: true, model: LORA_MODEL, weight: 0.75 };
const SETTINGS: GenerateSettings = { ...getDefaultGenerateSettings(), loras: [LORA] };
const ADAPTER = {
  models: {
    ModelSelect: () => null,
    getBaseColorPalette: () => 'gray',
    getBaseLabel: (base: string) => base,
    getImageUrl: () => '',
  },
} as unknown as GenerationUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async () => {
  const onCommit = vi.fn<(update: GenerateSettingsUpdate) => void>();
  const onCommitImmediate = vi.fn();

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const renderProject = (projectId: string) =>
    act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <GenerateConceptsContent
              loraModels={LORA_MODELS}
              projectId={projectId}
              selectedModel={MAIN_MODEL}
              settings={SETTINGS}
              onCommit={onCommit}
              onCommitImmediate={onCommitImmediate}
            />
          </GenerationUiProvider>
        </ChakraProvider>
      )
    );
  await renderProject('project-1');

  return { onCommit, onCommitImmediate, renderProject, row: host };
};

/** The settings a recorded commit produces from the rendered ones. */
const applied = (update: GenerateSettingsUpdate | undefined): GenerateLora | undefined =>
  (typeof update === 'function' ? update(SETTINGS) : { ...SETTINGS, ...update }).loras[0];

const stepWeight = (row: HTMLElement) =>
  act(() => {
    row
      .querySelector('[role="slider"]')
      ?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
  });

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateConceptsContent', () => {
  it('discards a concept menu when switching projects with the same concept', async () => {
    const { onCommit, onCommitImmediate, renderProject, row } = await render();

    await act(() => {
      row
        .querySelector('[data-list-primary]')
        ?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();

    await renderProject('project-2');

    expect(document.querySelector('[role="menu"][data-state="open"]')).toBeNull();
    await renderProject('project-1');
    expect(document.querySelector('[role="menu"][data-state="open"]')).toBeNull();
    expect(onCommit).not.toHaveBeenCalled();
    expect(onCommitImmediate).not.toHaveBeenCalled();
  });

  it('holds weight steps as a draft and commits them once the debounce settles', async () => {
    const { onCommit, row } = await render();

    await stepWeight(row);
    await stepWeight(row);

    expect(row.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('0.85');
    expect(onCommit).not.toHaveBeenCalled();

    await expect.poll(() => onCommit.mock.calls.length, { timeout: 1000 }).toBe(1);
    expect(applied(onCommit.mock.calls[0]?.[0])?.weight).toBe(0.85);
  });

  it('commits a pending weight when drafts are flushed', async () => {
    const { onCommit, row } = await render();

    await stepWeight(row);
    act(() => flushGenerateDrafts());

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(applied(onCommit.mock.calls[0]?.[0])?.weight).toBe(0.8);
  });

  it('commits toggles immediately rather than through the weight draft', async () => {
    const { onCommit, row } = await render();

    await act(() => row.querySelector<HTMLElement>('[data-scope="switch"][data-part="control"]')?.click());

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(applied(onCommit.mock.calls[0]?.[0])).toMatchObject({ isEnabled: false, weight: 0.75 });
  });
});
