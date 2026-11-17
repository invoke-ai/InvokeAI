import type { GenerateModelConfig, GenerateReferenceImage } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider, type GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { getModeLabelKey } from './referenceImageConfig';
import { ReferenceImageRow } from './ReferenceImageRow';

vi.mock('react-i18next', () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});

const ADAPTER = {
  models: {
    ModelSelect: () => <button type="button">IP-Adapter picker</button>,
    getBaseColorPalette: () => 'gray',
    getBaseLabel: (base: string) => base,
    getImageUrl: () => '',
  },
} as unknown as GenerationUiAdapter;

const ANIMA: GenerateModelConfig = { base: 'anima', key: 'anima', name: 'Anima', type: 'main', variant: 'anima_qwen3' };
const SDXL: GenerateModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };

const referenceImage = (base: string): GenerateReferenceImage => ({
  config: {
    beginEndStepPct: [0, 1],
    clipVisionModel: 'ViT-H',
    image: { original: { image: { height: 512, image_name: 'character.png', width: 512 } } },
    method: 'style',
    model: { base, key: `${base}-ip-adapter`, name: `${base} IP-Adapter`, type: 'ip_adapter' },
    type: 'ip_adapter',
    weight: 1.5,
  },
  id: 'ref-1',
  isEnabled: true,
});

const onPatch = vi.fn();
let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (selectedModel: GenerateModelConfig) => {
  onPatch.mockClear();
  host = document.createElement('div');
  host.style.width = '360px';
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GenerationUiProvider adapter={ADAPTER}>
          <ReferenceImageRow
            count={1}
            index={0}
            referenceImage={referenceImage(selectedModel.base)}
            selectedModel={selectedModel}
            onFindInGallery={vi.fn()}
            onMove={vi.fn()}
            onPatch={onPatch}
            onRemove={vi.fn()}
            onRename={vi.fn()}
            onUseSize={vi.fn()}
          />
        </GenerationUiProvider>
      </ChakraProvider>
    )
  );
  // The settings under the row hold the mode, the model picker and active steps.
  await act(() =>
    host?.querySelector<HTMLButtonElement>('button[aria-label="widgets.generate.referenceImageSettings"]')?.click()
  );

  return host;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('Anima IP-Adapter reference row', () => {
  it('offers strength and active steps, without the SD mode, style variant or CLIP Vision encoder', async () => {
    const card = await render(ANIMA);
    const text = card.textContent ?? '';

    expect(text).toContain('widgets.generate.strength');
    expect(text).not.toContain('widgets.generate.weight');
    expect(card.querySelector('[aria-label="widgets.generate.mode"]')).toBeNull();
    expect(text).not.toContain('widgets.generate.styleVariant');
    expect(text).not.toContain('ViT-H');
    expect(text).toContain('widgets.generate.activeStepsShort');
    expect(text).toContain('IP-Adapter picker');

    // The row's own control, ahead of the step range under it.
    const strength = card.querySelector('[role="slider"]');
    expect(strength?.getAttribute('aria-valuenow')).toBe('1.5');
    expect(strength?.getAttribute('aria-valuemin')).toBe('0');
    expect(strength?.getAttribute('aria-valuemax')).toBe('2');
  });

  it('clamps a typed negative strength to 0, the lowest the node accepts', async () => {
    const card = await render(ANIMA);
    const strength = card.querySelector<HTMLElement>('[role="slider"]')!;

    await act(async () => {
      strength.focus();
      await userEvent.keyboard('{Enter}');
    });
    await act(async () => {
      await userEvent.fill(card.querySelector<HTMLInputElement>('input[data-part="value"]')!, '-1');
      await userEvent.keyboard('{Enter}');
    });

    expect(onPatch).toHaveBeenLastCalledWith('ref-1', { config: expect.objectContaining({ weight: 0 }) });
  });

  it('summarizes the row by its adapter, not an SD mode', async () => {
    const card = await render(ANIMA);
    const summary = card.querySelector('[data-list-primary]')?.textContent ?? '';

    expect(summary).toContain('anima IP-Adapter');
    expect(summary).not.toContain(getModeLabelKey('style'));
  });

  it('keeps the SD controls for an SDXL reference', async () => {
    const text = (await render(SDXL)).textContent ?? '';

    const card = host!;

    expect(text).toContain('widgets.generate.weight');
    expect(card.querySelector('[aria-label="widgets.generate.mode"]')).not.toBeNull();
    expect(card.querySelector('[data-list-primary]')?.textContent).toContain(getModeLabelKey('style'));
    expect(text).toContain('widgets.generate.styleVariant');
    expect(text).toContain('ViT-H');
    expect(text).not.toContain('widgets.generate.strength');
  });
});
