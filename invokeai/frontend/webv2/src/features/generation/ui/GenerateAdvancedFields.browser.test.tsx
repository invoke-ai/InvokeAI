/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
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
    models: { getBaseLabel: (base: string) => ({ 'sd-1': 'SD 1.x', sdxl: 'SDXL' })[base] ?? base },
    sectionPreferences: { sectionsOpen: { advanced: true }, setSectionsOpen: vi.fn() },
  }),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' };
const krea2Model: MainModelConfig = {
  base: 'krea-2',
  format: 'diffusers',
  key: 'krea2',
  name: 'Krea-2 Turbo',
  type: 'main',
};
let host: HTMLDivElement | null = null;
let onCommitImmediate = vi.fn();
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
  const draft = createExternalStoreCore(settings);
  const onCommit = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));
  onCommitImmediate = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateAdvancedFields
            draft={draft}
            selectedModel={model}
            onCommit={onCommit}
            onCommitImmediate={onCommitImmediate}
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
  it('limits manual editing to ordered ratios while allowing equal cutoffs', async () => {
    const onCommit = await render(sd1Model, true, { hiDiffusionT1Ratio: 0.7, hiDiffusionT2Ratio: 0.2 });
    const t2 = sliderByLabel('HiDiffusion: T2 Ratio') as HTMLElement;
    await userEvent.click(t2);
    await userEvent.keyboard('{End}');
    expect(onCommit).toHaveBeenLastCalledWith({ hiDiffusionT2Ratio: 0.7 });
    expect(sliderByLabel('HiDiffusion: T1 Ratio')?.getAttribute('aria-valuenow')).toBe('0.7');
    const t1 = sliderByLabel('HiDiffusion: T1 Ratio') as HTMLElement;
    await userEvent.click(t1);
    await userEvent.keyboard('{Home}');
    expect(sliderByLabel('HiDiffusion: T1 Ratio')?.getAttribute('aria-valuenow')).toBe('0.7');
    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(sliderByLabel('HiDiffusion: T2 Ratio')?.getAttribute('aria-valuenow')).toBe('0.7');
  });

  it('restores automatic ratios and disables ratios when RAU-Net is off', async () => {
    const onCommit = await render(sd1Model, true, { hiDiffusionT1Ratio: 0, hiDiffusionT2Ratio: 0 });
    const automatic = switchByLabel('HiDiffusion: Automatic ratios');
    expect(automatic?.querySelector<HTMLInputElement>('input')?.checked).toBe(false);
    await settle(() => automatic?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onCommit).toHaveBeenCalledWith({
      hiDiffusionT1Ratio: null,
      hiDiffusionT2Ratio: null,
      hiDiffusionManualRatios: { t1: 0, t2: 0 },
    });
    await settle(() => root?.unmount());
    await render(sd1Model, true, { hiDiffusionRauNetEnabled: false, hiDiffusionT1Ratio: 0.4, hiDiffusionT2Ratio: 0 });
    expect(switchByLabel('HiDiffusion: Automatic ratios')?.querySelector<HTMLInputElement>('input')?.disabled).toBe(
      true
    );
    expect(sliderByLabel('HiDiffusion: T1 Ratio')?.getAttribute('aria-disabled')).toBe('true');
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
    expect(kvCacheInput()?.checked).toBe(false);
    expect(host?.textContent).toContain('Applies only with Edit references and an edit LoRA in Concepts.');
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

    expect(kvCacheInput()?.disabled).toBe(false);
    expect(kvCacheInput()?.checked).toBe(true);
    expect(host?.textContent).toContain('1 override');
  });
});

describe('GenerateAdvancedFields reset dots', () => {
  seedArchitectureCapabilities();

  const dot = (field: string, reset = 'Reset to default') =>
    host?.querySelector<HTMLElement>(`[data-modified-dot][aria-label="${reset}: ${field}"]`) ?? null;

  it('resets VAE precision from its dot, not a separate button, and hands focus to the select', async () => {
    const defaults = getDefaultGenerateSettings(sd1Model);
    const onCommit = await render(sd1Model, false, {
      vaePrecision: defaults.vaePrecision === 'fp16' ? 'fp32' : 'fp16',
    });

    expect(host?.querySelector('button[aria-label="Use model default VAE precision"]')).toBeNull();
    await settle(() => dot('VAE precision', 'Reset to SD 1.x default')?.click());

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({ vaePrecision: defaults.vaePrecision });
    expect(dot('VAE precision', 'Reset to SD 1.x default')).toBeNull();
    await vi.waitFor(() => expect(document.activeElement).toBe(selectByLabel('VAE precision')));
  });

  it('turns manual ratios back to automatic from the dot, keeping the manual values for later', async () => {
    const onCommit = await render(sd1Model, true, {
      hiDiffusionManualRatios: { t1: 0.7, t2: 0.2 },
      hiDiffusionT1Ratio: 0.7,
      hiDiffusionT2Ratio: 0.2,
    });

    await settle(() => dot('HiDiffusion: Automatic ratios')?.click());

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({
      hiDiffusionManualRatios: { t1: 0.7, t2: 0.2 },
      hiDiffusionT1Ratio: null,
      hiDiffusionT2Ratio: null,
    });
    expect(dot('HiDiffusion: Automatic ratios')).toBeNull();
  });
});
