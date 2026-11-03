/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GenerateAdvancedFields } from './GenerateAdvancedFields';

vi.mock('./GenerationUiContext', () => ({
  GenerationModelSelect: () => null,
  useGenerationUi: () => ({
    sectionPreferences: { sectionsOpen: { advanced: true }, setSectionOpen: vi.fn() },
  }),
}));

// The real catalog, so a mistyped key shows up as the key and fails the label lookups below.
const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' };
const sd2Model: MainModelConfig = { base: 'sd-2', key: 'sd2', name: 'SD 2', type: 'main' };
const fluxModel: MainModelConfig = { base: 'flux', key: 'flux', name: 'FLUX.1 dev', type: 'main' };
const qwenImageModel: MainModelConfig = {
  base: 'qwen-image',
  format: 'diffusers',
  key: 'qwen-image',
  name: 'Qwen Image',
  type: 'main',
};
const krea2Model: MainModelConfig = {
  base: 'krea-2',
  format: 'diffusers',
  key: 'krea2',
  name: 'Krea-2 Turbo',
  type: 'main',
};
let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const render = async (
  model: MainModelConfig,
  hiDiffusionEnabled: boolean,
  overrides: Partial<GenerateSettings> = {}
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const settings = { ...getDefaultGenerateSettings(model), hiDiffusionEnabled, ...overrides };
  const onCommit = vi.fn();

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateAdvancedFields
            draft={createExternalStoreCore(settings)}
            selectedModel={model}
            onCommit={onCommit}
            onCommitImmediate={vi.fn()}
          />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return onCommit;
};

const switchByLabel = (label: string): HTMLElement | null =>
  [...(host?.querySelectorAll<HTMLElement>('[data-scope="switch"][data-part="root"]') ?? [])].find(
    (element) => element.textContent === label
  ) ?? null;

/** Sliders and select triggers are named by `aria-labelledby`, so resolve them through the label element. */
const labelledBy = (selector: string, label: string): Element | null =>
  [...(host?.querySelectorAll<HTMLElement>(selector) ?? [])].find((control) => {
    const labelId = control.getAttribute('aria-labelledby');

    return labelId !== null && host?.querySelector(`#${CSS.escape(labelId)}`)?.textContent === label;
  }) ?? null;

const sliderByLabel = (label: string): Element | null => labelledBy('[role="slider"]', label);

const selectByLabel = (label: string): Element | null =>
  labelledBy('[data-scope="select"][data-part="trigger"]', label);

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateAdvancedFields HiDiffusion controls', () => {
  it('shows HiDiffusion for SD1 and disables dependent controls while it is off', async () => {
    const onCommit = await render(sd1Model, false);
    const main = switchByLabel('HiDiffusion');
    const rauNet = switchByLabel('HiDiffusion: RAU-Net');

    expect(main).not.toBeNull();
    expect(rauNet?.querySelector<HTMLInputElement>('input')?.disabled).toBe(true);
    expect(sliderByLabel('HiDiffusion: T1 Ratio')?.getAttribute('aria-disabled')).toBe('true');

    // A disabled row's label is inert; the enabled row's label toggles its own switch.
    await act(() => userEvent.click(rauNet!.querySelector<HTMLElement>('[data-part="label"]')!, { force: true }));
    expect(onCommit).not.toHaveBeenCalled();

    await act(() => userEvent.click(main!.querySelector<HTMLElement>('[data-part="label"]')!));
    expect(onCommit).toHaveBeenCalledWith({ hiDiffusionEnabled: true });
    expect(onCommit).toHaveBeenCalledTimes(1);
  });

  it('omits HiDiffusion controls for SD2', async () => {
    await render(sd2Model, true);

    expect(switchByLabel('HiDiffusion')).toBeNull();
    expect(host?.textContent).not.toContain('HiDiffusion: T1 Ratio');
  });
});

describe('GenerateAdvancedFields PiD control', () => {
  // FLUX has no other advanced field, so the section must not hide itself while PiD is supported.
  it('shows the PiD mode for a base whose only advanced field is PiD', async () => {
    await render(fluxModel, false);

    expect(selectByLabel('PiD Decode')).not.toBeNull();
  });

  it('omits the PiD mode for a base without PiD decoders', async () => {
    await render(sd1Model, false);

    expect(selectByLabel('PiD Decode')).toBeNull();
  });
});

describe('GenerateAdvancedFields encoder quantization', () => {
  it('offers the Qwen VL encoder quantization for Qwen Image only', async () => {
    await render(qwenImageModel, false);

    expect(selectByLabel('Encoder quantization')).not.toBeNull();
    expect(selectByLabel('Encoder quantization')?.textContent).toContain('None (bf16)');

    await settle(() => root?.unmount());
    await render(fluxModel, false);

    expect(selectByLabel('Encoder quantization')).toBeNull();
  });
});

describe('GenerateAdvancedFields Krea-2 reference K/V cache', () => {
  const editReference = {
    config: {
      image: { original: { image: { height: 512, image_name: 'pose.png', width: 512 } } },
      type: 'krea2_edit_reference_image' as const,
    },
    id: 'pose',
    isEnabled: true,
  };
  const poseLora = {
    isEnabled: true,
    model: { base: 'krea-2', key: 'pose-lora', name: 'Pose LoRA', type: 'lora' as const },
    weight: 1,
  };
  const kvCacheInput = () => switchByLabel('Reference K/V cache')?.querySelector<HTMLInputElement>('input') ?? null;

  // Krea-2 has no other advanced field, so the section must not hide itself.
  it('offers the switch with edit references and a LoRA, and commits it', async () => {
    const onCommit = await render(krea2Model, false, { loras: [poseLora], referenceImages: [editReference] });

    expect(kvCacheInput()?.disabled).toBe(false);
    expect(host?.textContent).toContain('For edit LoRAs trained with isolated references.');

    await settle(() =>
      switchByLabel('Reference K/V cache')?.querySelector<HTMLElement>('[data-part="control"]')?.click()
    );
    expect(onCommit).toHaveBeenCalledWith({ krea2ReferenceKvCache: true });
  });

  it('disables the switch, shows it off and says why without edit references or a LoRA, keeping the stored value', async () => {
    await render(krea2Model, false, { krea2ReferenceKvCache: true, loras: [poseLora] });

    expect(kvCacheInput()?.disabled).toBe(true);
    // Shown off while it is not sent.
    expect(kvCacheInput()?.checked).toBe(false);
    expect(host?.textContent).toContain('Applies only with Edit references and an edit LoRA in Concepts.');
    // Not sent, so not counted as an override.
    expect(host?.textContent).not.toContain('1 override');

    await settle(() => root?.unmount());
    await render(krea2Model, false, {
      krea2ReferenceKvCache: true,
      loras: [{ ...poseLora, isEnabled: false }],
      referenceImages: [editReference],
    });

    expect(kvCacheInput()?.disabled).toBe(true);
    expect(host?.textContent).toContain('Applies only with an edit LoRA enabled in Concepts.');
    expect(host?.textContent).not.toContain('1 override');

    await settle(() => root?.unmount());
    await render(krea2Model, false, {
      krea2ReferenceKvCache: true,
      loras: [poseLora],
      referenceImages: [editReference],
    });

    // The stored value is back once the references and the LoRA are.
    expect(kvCacheInput()?.disabled).toBe(false);
    expect(kvCacheInput()?.checked).toBe(true);
    expect(host?.textContent).toContain('1 override');
  });

  it('omits the reference K/V cache for other families', async () => {
    await render(qwenImageModel, false);

    expect(switchByLabel('Reference K/V cache')).toBeNull();
  });
});
