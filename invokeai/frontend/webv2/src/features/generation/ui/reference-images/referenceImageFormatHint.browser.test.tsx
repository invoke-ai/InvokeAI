import type {
  ComponentModelConfig,
  GenerateModelConfig,
  GenerateReferenceImage,
} from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { type GenerationUiAdapter, GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GenerateReferenceImagesContent } from './GenerateReferenceImagesSection';

/** A model whose encoder has no vision tower explains why the reference section is gone; other models stay silent. */
const i18n = i18next.createInstance();
// The shipped copy, so a change to the hint's wording or placeholders fails here.
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

seedArchitectureCapabilities();

const ADAPTER = {
  gallery: { findImage: vi.fn() },
  models: { getBaseLabel: (base: string) => (base === 'qwen-image-2-1' ? 'Qwen-Image-2.1' : base) },
} as unknown as GenerationUiAdapter;

const QWEN21_GGUF: GenerateModelConfig = {
  base: 'qwen-image-2-1',
  format: 'gguf_quantized',
  key: 'qwen21-q4',
  name: 'Qwen-Image-2.1 Q4',
  type: 'main',
};
const QWEN21_CHECKPOINT: GenerateModelConfig = { ...QWEN21_GGUF, format: 'checkpoint', key: 'qwen21-int8' };
const ENCODER_8B = { base: 'any', key: 'enc', name: 'Qwen3-VL 8B', type: 'qwen3_vl_encoder', variant: 'qwen3_vl_8b' };
const NO_MODELS: [] = [];
const HINT =
  'Reference images need a Qwen3-VL 8B encoder that is not GGUF. Select one under Components (e.g. Qwen3-VL 8B Encoder), or use a Diffusers Qwen-Image-2.1 model.';
const COGVIEW4: GenerateModelConfig = { base: 'cogview4', key: 'cogview4', name: 'CogView4', type: 'main' };
const QWEN21_REFERENCE: GenerateReferenceImage = {
  config: {
    image: { original: { image: { height: 512, image_name: 'edit.png', width: 512 } } },
    type: 'qwen_image_2_1_reference_image',
  },
  id: 'edit',
  isEnabled: true,
};

let host: HTMLDivElement | null = null;
// DndContext renders its own screen-reader instructions, so assertions read only the section's output.
const section = (): HTMLElement | null | undefined => host?.querySelector('[data-testid="section"]');
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (
  selectedModel: GenerateModelConfig,
  referenceImages: GenerateReferenceImage[] = [],
  qwen3VLEncoderModel: ComponentModelConfig | null = null
) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <DndContext>
              <div data-testid="section">
                <GenerateReferenceImagesContent
                  models={NO_MODELS}
                  qwen3VLEncoderModel={qwen3VLEncoderModel}
                  referenceImages={referenceImages}
                  selectedModel={selectedModel}
                  onCommit={vi.fn()}
                  onCommitImmediate={vi.fn()}
                />
              </div>
            </DndContext>
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('reference image vision tower hint', () => {
  it('names the encoders that read references where a GGUF Qwen-Image-2.1 model would show them', async () => {
    await render(QWEN21_GGUF);

    expect(section()?.textContent).toBe(HINT);
    expect(section()?.querySelector('button')).toBeNull();
  });

  it('keeps the hint next to the clear action for references the model cannot read', async () => {
    await render(QWEN21_GGUF, [QWEN21_REFERENCE]);

    expect(section()?.textContent).toContain(HINT);
    expect(section()?.querySelector('button')?.textContent).toBe('Remove all');
  });

  it('offers references for a single-file model once a safetensors encoder is selected', async () => {
    await render(QWEN21_CHECKPOINT, [], { ...ENCODER_8B, format: 'checkpoint' });

    expect(section()?.textContent).not.toContain(HINT);
    expect(section()?.querySelector('button')).not.toBeNull();
  });

  it('keeps the hint when the selected encoder is GGUF', async () => {
    await render(QWEN21_CHECKPOINT, [], { ...ENCODER_8B, format: 'gguf_quantized' });

    expect(section()?.textContent).toBe(HINT);
  });

  it('renders nothing for a model whose base takes no references in any format', async () => {
    await render(COGVIEW4);

    expect(section()?.textContent).toBe('');
  });
});
