/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateConceptsContent } from './GenerateConceptsSection';
import { GenerationUiProvider, type GenerationUiAdapter } from './GenerationUiContext';

vi.mock('react-i18next', () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});

seedArchitectureCapabilities();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const sdxl: GenerateModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
const sd3: GenerateModelConfig = { base: 'sd-3', key: 'sd3', name: 'SD3', type: 'main' };
const inkLora: LoraModelConfig = { base: 'sdxl', key: 'ink', name: 'Ink', type: 'lora' };

const adapter = {
  models: {
    ModelSelect: () => <div data-testid="concept-picker" />,
    getBaseColorPalette: () => 'gray',
    getBaseLabel: (base: string) => base,
  },
} as unknown as GenerationUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const renderConcepts = async (selectedModel: GenerateModelConfig, loras: GenerateSettings['loras']) => {
  const onCommitImmediate = vi.fn();

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <GenerationUiProvider adapter={adapter}>
          <GenerateConceptsContent
            loraModels={[inkLora]}
            projectId="project-1"
            selectedModel={selectedModel}
            settings={{ ...getDefaultGenerateSettings(sdxl as never), loras }}
            onCommit={vi.fn()}
            onCommitImmediate={onCommitImmediate}
          />
        </GenerationUiProvider>
      </ChakraProvider>
    );
  });

  return { host, onCommitImmediate };
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateConceptsContent', () => {
  it('offers the concept picker for a model family that loads LoRAs', async () => {
    const { host } = await renderConcepts(sdxl, []);

    expect(host.querySelector('[data-testid="concept-picker"]')).not.toBeNull();
    expect(host.textContent).toContain('widgets.generate.addConceptsHelp');
    expect(host.textContent).not.toContain('widgets.generate.conceptsUnsupported');
  });

  it('explains instead of offering an always-empty picker when the model cannot use concepts', async () => {
    const { host } = await renderConcepts(sd3, []);

    expect(host.querySelector('[data-testid="concept-picker"]')).toBeNull();
    expect(host.textContent).toContain('widgets.generate.conceptsUnsupported');
    expect(host.textContent).not.toContain('widgets.generate.addConceptsHelp');
  });

  it('keeps concepts from another model removable when the model cannot use concepts', async () => {
    const { host, onCommitImmediate } = await renderConcepts(sd3, [{ isEnabled: true, model: inkLora, weight: 0.8 }]);

    expect(host.textContent).toContain('Ink');
    expect(host.textContent).toContain('widgets.generate.incompatible');

    const remove = host.querySelector<HTMLButtonElement>('button[aria-label="widgets.generate.removeConceptNamed"]');

    expect(remove).not.toBeNull();
    await act(() => {
      remove?.click();
    });

    expect(onCommitImmediate).toHaveBeenCalledWith({ loras: [] });
  });

  it('marks a same-family concept incompatible when the model cannot use concepts', async () => {
    const sd3Lora: LoraModelConfig = { base: 'sd-3', key: 'sd3-lora', name: 'Glow', type: 'lora' };
    const { host } = await renderConcepts(sd3, [{ isEnabled: true, model: sd3Lora, weight: 0.8 }]);

    expect(host.textContent).toContain('widgets.generate.incompatible');
  });
});
