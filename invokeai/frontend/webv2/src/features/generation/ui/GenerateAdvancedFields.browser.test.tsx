/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@features/generation/core/settings';
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
    sectionPreferences: { sectionsOpen: { advanced: true }, setSectionsOpen: vi.fn() },
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
const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
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

  it('shows HiDiffusion for SD1 and disables dependent controls while it is off', async () => {
    const onCommit = await render(sd1Model, false, { hiDiffusionT1Ratio: 0.4, hiDiffusionT2Ratio: 0 });
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

  it('defaults to automatic ratios and enables manual overrides on request', async () => {
    const onCommit = await render(sd1Model, true);
    const automatic = switchByLabel('HiDiffusion: Automatic ratios');
    expect(automatic?.querySelector<HTMLInputElement>('input')?.checked).toBe(true);
    expect(sliderByLabel('HiDiffusion: T1 Ratio')?.getAttribute('aria-disabled')).toBe('true');
    expect(host?.textContent).toContain('Automatic');
    await settle(() => automatic?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onCommit).toHaveBeenCalledWith({ hiDiffusionT1Ratio: 0.4, hiDiffusionT2Ratio: 0 });
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

  const ratioDot = (ratio: number) =>
    host?.querySelector<HTMLElement>(`[data-modified-dot][aria-label$="HiDiffusion: T${ratio} Ratio"]`) ?? null;
  const ratioMenuReset = async (ratio: number) => {
    await settle(() =>
      sliderByLabel(`HiDiffusion: T${ratio} Ratio`)?.dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true })
      )
    );
    return document.querySelector<HTMLElement>('[data-scope="menu"][data-part="item"][data-value="reset"]');
  };

  it.each([1, 2].flatMap((ratio) => ['menu', 'dot'].map((action) => ({ action, ratio }))))(
    'resets only a manual T$ratio through its $action',
    async ({ action, ratio }) => {
      const onCommit = await render(sd1Model, true, {
        hiDiffusionManualRatios: { t1: 0.8, t2: 0.3 },
        hiDiffusionT1Ratio: 0.7,
        hiDiffusionT2Ratio: 0.2,
      });
      const label = `HiDiffusion: T${ratio} Ratio`;

      expect(ratioDot(ratio)).not.toBeNull();
      if (action === 'menu') {
        const reset = await ratioMenuReset(ratio);
        expect(reset).not.toBeNull();
        await settle(() => reset?.click());
      } else {
        await settle(() => ratioDot(ratio)?.click());
      }

      expect(onCommit).toHaveBeenCalledExactlyOnceWith(
        ratio === 1 ? { hiDiffusionT1Ratio: 0.4 } : { hiDiffusionT2Ratio: 0 }
      );
      expect(sliderByLabel(label)?.getAttribute('aria-valuenow')).toBe(ratio === 1 ? '0.4' : '0');
      expect(sliderByLabel(`HiDiffusion: T${ratio === 1 ? 2 : 1} Ratio`)?.getAttribute('aria-valuenow')).toBe(
        ratio === 1 ? '0.2' : '0.7'
      );
      expect(ratioDot(ratio)).toBeNull();
      expect((await ratioMenuReset(ratio))?.hasAttribute('data-disabled')).toBe(true);
    }
  );

  // A disabled ratio is not sent, so it shows no dot and its menu offers no reset, whatever value it keeps.
  it.each([
    ['automatic ratios', { hiDiffusionT1Ratio: null, hiDiffusionT2Ratio: null }, [1, 2]],
    ['an automatic T1 beside a manual T2', { hiDiffusionT1Ratio: null, hiDiffusionT2Ratio: 0.2 }, [1]],
    ['RAU-Net off', { hiDiffusionRauNetEnabled: false, hiDiffusionT1Ratio: 0.7, hiDiffusionT2Ratio: 0.2 }, [1, 2]],
    ['HiDiffusion off', { hiDiffusionEnabled: false, hiDiffusionT1Ratio: 0.7, hiDiffusionT2Ratio: 0.2 }, [1, 2]],
  ] as const)('marks no disabled ratio with %s', async (_case, overrides, disabledRatios) => {
    await render(sd1Model, true, { hiDiffusionManualRatios: { t1: 0.8, t2: 0.3 }, ...overrides });

    for (const ratio of disabledRatios) {
      expect(sliderByLabel(`HiDiffusion: T${ratio} Ratio`)?.getAttribute('aria-disabled')).toBe('true');
      expect(ratioDot(ratio)).toBeNull();
      expect((await ratioMenuReset(ratio))?.hasAttribute('data-disabled')).toBe(true);
      await settle(() => document.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));
    }
  });

  it.each(['menu', 'dot'])('keeps the manual pair ordered when resetting T1 through its %s', async (action) => {
    const onCommit = await render(sd1Model, true, {
      hiDiffusionT1Ratio: 0.8,
      hiDiffusionT2Ratio: 0.6,
      hiDiffusionManualRatios: { t1: 0.8, t2: 0.6 },
    });
    const label = 'HiDiffusion: T1 Ratio';

    expect(ratioDot(1)).not.toBeNull();
    if (action === 'menu') {
      const reset = await ratioMenuReset(1);
      expect(reset).not.toBeNull();
      await settle(() => reset?.click());
    } else {
      await settle(() => ratioDot(1)?.click());
    }

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({ hiDiffusionT1Ratio: 0.6 });
    expect(sliderByLabel(label)?.getAttribute('aria-valuenow')).toBe('0.6');
    expect(sliderByLabel('HiDiffusion: T2 Ratio')?.getAttribute('aria-valuenow')).toBe('0.6');
    expect(ratioDot(1)).toBeNull();
    expect((await ratioMenuReset(1))?.hasAttribute('data-disabled')).toBe(true);
  });

  it.each([1, 2])(
    'keeps a recalled automatic T%s slider disabled without changing its manual companion',
    async (ratio) => {
      await render(sd1Model, true, {
        hiDiffusionT1Ratio: ratio === 1 ? null : 0.7,
        hiDiffusionT2Ratio: ratio === 2 ? null : 0.2,
      });
      expect(sliderByLabel(`HiDiffusion: T${ratio} Ratio`)?.getAttribute('aria-disabled')).toBe('true');
      expect(host?.textContent).toContain('Automatic');
      expect(sliderByLabel(`HiDiffusion: T${ratio === 1 ? 2 : 1} Ratio`)?.getAttribute('aria-disabled')).not.toBe(
        'true'
      );
      expect(switchByLabel('HiDiffusion: Automatic ratios')?.querySelector<HTMLInputElement>('input')?.checked).toBe(
        false
      );
    }
  );

  it.each([
    { t1: 0.7, t2: 0.2 },
    { t1: 0, t2: 0 },
  ])(
    'retains manual slider positions through automatic mode and a saved-settings remount ($t1, $t2)',
    async (manual) => {
      const initial = {
        ...getDefaultGenerateSettings(sd1Model),
        hiDiffusionEnabled: true,
        hiDiffusionT1Ratio: manual.t1,
        hiDiffusionT2Ratio: manual.t2,
      };
      const onCommit = await render(sd1Model, true, initial);
      await settle(() =>
        switchByLabel('HiDiffusion: Automatic ratios')?.querySelector<HTMLElement>('[data-part="control"]')?.click()
      );
      const automaticPatch = onCommit.mock.lastCall?.[0];
      expect(automaticPatch).toMatchObject({
        hiDiffusionT1Ratio: null,
        hiDiffusionT2Ratio: null,
        hiDiffusionManualRatios: manual,
      });
      for (const [ratio, value] of [
        [1, manual.t1],
        [2, manual.t2],
      ]) {
        const slider = sliderByLabel(`HiDiffusion: T${ratio} Ratio`);
        expect(slider?.getAttribute('aria-disabled')).toBe('true');
        expect(slider?.getAttribute('aria-valuenow')).toBe(String(value));
      }
      const saved = normalizeGenerateSettings(JSON.parse(JSON.stringify({ ...initial, ...automaticPatch })));
      expect(saved).not.toBeNull();
      await settle(() => root?.unmount());
      host?.remove();
      const restoredCommit = await render(sd1Model, true, saved!);
      for (const [ratio, value] of [
        [1, manual.t1],
        [2, manual.t2],
      ]) {
        expect(sliderByLabel(`HiDiffusion: T${ratio} Ratio`)?.getAttribute('aria-valuenow')).toBe(String(value));
      }
      await settle(() =>
        switchByLabel('HiDiffusion: Automatic ratios')?.querySelector<HTMLElement>('[data-part="control"]')?.click()
      );
      expect(restoredCommit).toHaveBeenCalledWith({ hiDiffusionT1Ratio: manual.t1, hiDiffusionT2Ratio: manual.t2 });
      expect(sliderByLabel('HiDiffusion: T1 Ratio')?.getAttribute('aria-disabled')).not.toBe('true');
      expect(sliderByLabel('HiDiffusion: T2 Ratio')?.getAttribute('aria-disabled')).not.toBe('true');
    }
  );

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

  const kvCacheDot = () =>
    host?.querySelector<HTMLElement>('[data-modified-dot][aria-label="Reset to model default: Reference K/V cache"]') ??
    null;

  it('marks the switch with the dot while it applies and is on, and resets it', async () => {
    const onCommit = await render(krea2Model, false, {
      krea2ReferenceKvCache: true,
      loras: [poseLora],
      referenceImages: [editReference],
    });

    expect(kvCacheDot()).not.toBeNull();
    await settle(() => kvCacheDot()?.click());

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({ krea2ReferenceKvCache: false });
    expect(kvCacheDot()).toBeNull();
  });

  it('shows no dot while the switch does not apply, even with the stored value on', async () => {
    await render(krea2Model, false, { krea2ReferenceKvCache: true });

    expect(kvCacheInput()?.disabled).toBe(true);
    expect(kvCacheDot()).toBeNull();
  });

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

describe('GenerateAdvancedFields alignment', () => {
  // A field that pads itself sits inside the others' edges; the section owns the inset.
  it.each([
    ['SD 1.5', sd1Model, {}],
    ['Qwen Image', qwenImageModel, {}],
    ['Krea-2', krea2Model, {}],
  ] as const)('starts every %s control on the same left edge', async (_name, model, overrides) => {
    await render(model, true, overrides);
    await settle(() => host?.querySelector<HTMLElement>('.generate-section [data-part="trigger"]')?.click());

    // The controls themselves: a wrapper that pads its content still spans the full width.
    const controls = [
      ...(host?.querySelectorAll<HTMLElement>(
        '[data-part="content"] :is([data-scope="select"][data-part="trigger"], [data-scope="scrubber"][data-part="root"], [data-scope="switch"][data-part="label"])'
      ) ?? []),
    ]
      .map((control) => control.getBoundingClientRect())
      .filter((rect) => rect.width > 0);

    expect(controls.length).toBeGreaterThan(0);
    expect(new Set(controls.map((rect) => Math.round(rect.left))).size).toBe(1);
  });
});

describe('GenerateAdvancedFields reset dots', () => {
  // VAE, VAE precision and seamless tiling show per the architecture capabilities.
  seedArchitectureCapabilities();

  const dot = (field: string) =>
    host?.querySelector<HTMLElement>(`[data-modified-dot][aria-label="Reset to model default: ${field}"]`) ?? null;

  it('resets VAE precision from its dot, not a separate button, and hands focus to the select', async () => {
    const defaults = getDefaultGenerateSettings(sd1Model);
    const onCommit = await render(sd1Model, false, {
      vaePrecision: defaults.vaePrecision === 'fp16' ? 'fp32' : 'fp16',
    });

    expect(host?.querySelector('button[aria-label="Use model default VAE precision"]')).toBeNull();
    await settle(() => dot('VAE precision')?.click());

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({ vaePrecision: defaults.vaePrecision });
    expect(dot('VAE precision')).toBeNull();
    await vi.waitFor(() => expect(document.activeElement).toBe(selectByLabel('VAE precision')));
  });

  it('resets a chosen VAE to the bundled one from its dot', async () => {
    const vae = { base: 'sd-1', key: 'vae-1', name: 'Custom VAE', type: 'vae' } as unknown as GenerateSettings['vae'];
    await render(sd1Model, false, { vae });

    expect(host?.querySelector('button[aria-label="Use model default VAE"]')).toBeNull();
    await settle(() => dot('VAE')?.click());

    expect(onCommitImmediate).toHaveBeenCalledExactlyOnceWith({ vae: null });
    expect(dot('VAE')).toBeNull();
  });

  it.each([
    ['HiDiffusion', sd1Model, { hiDiffusionEnabled: true }, { hiDiffusionEnabled: false }, false],
    [
      'HiDiffusion: RAU-Net',
      sd1Model,
      { hiDiffusionEnabled: true, hiDiffusionRauNetEnabled: false },
      { hiDiffusionRauNetEnabled: true },
      true,
    ],
    [
      'HiDiffusion: Window Attention',
      sd1Model,
      { hiDiffusionEnabled: true, hiDiffusionWindowAttentionEnabled: false },
      { hiDiffusionWindowAttentionEnabled: true },
      true,
    ],
    ['Color compensation', sdxlModel, { colorCompensation: true }, { colorCompensation: false }, false],
    ['X axis', sd1Model, { seamlessXAxis: true }, { seamlessXAxis: false }, false],
    ['Y axis', sd1Model, { seamlessYAxis: true }, { seamlessYAxis: false }, false],
  ] as const)(
    'marks the %s switch off its default and resets it from the dot',
    async (label, model, changed, reset, on) => {
      const onCommit = await render(model, false, changed);
      const input = () => switchByLabel(label)?.querySelector<HTMLInputElement>('input') ?? null;

      expect(dot(label)).not.toBeNull();
      await settle(() => dot(label)?.click());

      expect(onCommit).toHaveBeenCalledExactlyOnceWith(reset);
      expect(input()?.checked).toBe(on);
      expect(dot(label)).toBeNull();
      await vi.waitFor(() => expect(document.activeElement).toBe(input()));
    }
  );

  it('shows no dot on a switch at its default or one that is not sent', async () => {
    // HiDiffusion off: its sub-switches are disabled, so even changed ones are not marked.
    await render(sd1Model, false, { hiDiffusionRauNetEnabled: false });

    expect(dot('HiDiffusion')).toBeNull();
    expect(dot('X axis')).toBeNull();
    expect(dot('HiDiffusion: RAU-Net')).toBeNull();
  });

  it('keeps a switch row dot and its target in the gutter, centred on the first line, clear of the label', async () => {
    await render(sd1Model, true);
    const button = dot('HiDiffusion')!;
    const label = switchByLabel('HiDiffusion')!.querySelector<HTMLElement>('[data-part="label"]')!;
    const control = switchByLabel('HiDiffusion')!.querySelector<HTMLElement>('[data-part="control"]')!;
    const rowLeft = label.getBoundingClientRect().left;
    const buttonRect = button.getBoundingClientRect();
    const dotRect = () => button.querySelector<HTMLElement>('[data-part="dot"]')!.getBoundingClientRect();
    const controlRect = control.getBoundingClientRect();

    expect(Math.abs((dotRect().top + dotRect().bottom) / 2 - (controlRect.top + controlRect.bottom) / 2)).toBeLessThan(
      0.5
    );
    // The whole target sits in the 8px gutter, so a click on the label's first letter toggles rather than resets.
    expect(buttonRect.left).toBeGreaterThanOrEqual(rowLeft - 8);
    expect(buttonRect.right).toBeLessThanOrEqual(rowLeft);
    expect(rowLeft - dotRect().right).toBe(3);

    await userEvent.hover(button);
    await vi.waitFor(() => expect(Math.round(dotRect().width)).toBe(6));
    expect(dotRect().left).toBeGreaterThanOrEqual(rowLeft - 8);
    expect(dotRect().right).toBeLessThanOrEqual(rowLeft - 2);
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

  it('resets PiD steps to the model default from its dot', async () => {
    const defaults = getDefaultGenerateSettings(qwenImageModel);
    const onCommit = await render(qwenImageModel, false, { pidMode: 'fit', pidSteps: defaults.pidSteps + 1 });

    await settle(() => dot('PiD steps')?.click());

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({ pidSteps: defaults.pidSteps });
    expect(dot('PiD steps')).toBeNull();
    await vi.waitFor(() => expect(document.activeElement).toBe(sliderByLabel('PiD steps')));
  });

  it('counts only what the section shows: a hidden VAE precision is no override', async () => {
    const defaults = getDefaultGenerateSettings(qwenImageModel);
    await render(qwenImageModel, false, { vaePrecision: defaults.vaePrecision === 'fp16' ? 'fp32' : 'fp16' });

    expect(selectByLabel('VAE precision')).toBeNull();
    expect(host?.textContent).not.toContain('override');
    expect(host?.querySelector('.generate-section [data-section-changed]')).toBeNull();
  });

  it('resets PiD and encoder quantization to the model defaults from their dots', async () => {
    const defaults = getDefaultGenerateSettings(qwenImageModel);
    const onCommit = await render(qwenImageModel, false, { pidMode: 'fit', qwenImageEncoderQuantization: 'int8' });

    await settle(() => dot('Encoder quantization')?.click());
    expect(onCommit).toHaveBeenCalledExactlyOnceWith({
      qwenImageEncoderQuantization: defaults.qwenImageEncoderQuantization,
    });

    await settle(() => dot('PiD Decode')?.click());
    expect(onCommitImmediate).toHaveBeenCalledExactlyOnceWith({ pidMode: defaults.pidMode });
    expect(dot('PiD Decode')).toBeNull();
  });
});
