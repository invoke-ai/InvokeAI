/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateModelConfig, GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { createExternalStoreCore, type ExternalStoreCore } from '@platform/state/externalStoreCore';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { CanvasDenoisingStrengthProps } from './GenerationUiContext';

import { GenerateRenderSection } from './GenerateRenderSection';

const seedHistory = [
  { seed: 777, thumbnailUrl: null },
  { seed: 888, thumbnailUrl: null },
];

const harness = vi.hoisted(() => ({ invocationSourceId: 'generate' }));

vi.mock('./GenerationUiContext', () => ({
  useGenerationQueueInsights: (select: (insights: unknown) => unknown) => select({ secondsPerRun: null, seedHistory }),
  useGenerationUi: () => ({
    // Stands in for the canvas's strength slot, which lays out the section around its badge and field.
    CanvasDenoisingStrength: ({ children }: CanvasDenoisingStrengthProps) =>
      children({
        badges: <span data-testid="strength-badge">75%</span>,
        field: <div data-testid="strength-field" />,
      }),
    project: { invocationSourceId: harness.invocationSourceId },
    sectionPreferences: { sectionsOpen: { render: true }, setSectionOpen: vi.fn() },
  }),
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: {
          newSeed: 'New seed',
          scrubber: { editValue: 'Edit {{label}}: {{value}}' },
          seed: 'Seed',
          seedMode: {
            decrement: 'Decrement',
            decrementDescription: 'Use successive seeds, decreasing by 1.',
            fixed: 'Fixed',
            fixedDescription: 'Reuse the entered seed.',
            increment: 'Increment',
            incrementDescription: 'Use successive seeds, increasing by 1.',
            label: 'Seed mode',
            random: 'Random',
            randomDescription: 'Choose a fresh starting seed for each submission.',
          },
          seedNextBatch: 'Next batch: {{seed}}',
          seedNextBatchRange: 'Next batch: {{first}} → {{last}}',
        },
        widgets: {
          generate: {
            // Retain duplicate labels in fixtures so tests expose real naming collisions.
            ideogram4ColorHelp: 'Comma-separated color terms folded into the caption.',
            ideogram4ColorPalette: 'Color palette',
            ideogram4GuidanceScale: 'Guidance',
            ideogram4Mu: 'Mu',
            ideogram4MuHelp:
              'Timestep shift. Higher values spend more of the schedule at high noise. Set by the sampler preset unless overridden.',
            ideogram4PresetDerived: 'Set by the sampler preset unless overridden.',
            ideogram4SamplerPreset: 'Sampler preset',
            ideogram4Steps: 'Steps',
            dypeExponent: 'DyPE exponent (λt)',
            dypePreset: 'DyPE',
            dypePreset4k: '4K optimized',
            dypePresetArea: 'Area (auto)',
            dypePresetAuto: 'Auto (above 1536px)',
            dypePresetManual: 'Manual',
            dypePresetOff: 'Off',
            dypeScale: 'DyPE scale (λs)',
            mossImageSkipLayers: 'Skip layers (negative branch)',
            mossImageSkipLayersCfgOff: 'Takes effect with CFG above 1, which runs the negative pass.',
            mossImageSkipLayersHelp:
              'Transformer blocks the negative pass skips, comma-separated from 0 to {{max}}. ' +
              'Default: {{defaultLayers}}. Empty skips none.',
            mossImageSkipLayersInvalid: 'Enter comma-separated whole numbers from 0 to {{max}}.',
            seedVariance: 'Seed variance',
            seedVarianceHelp: 'Perturbs conditioning so nearby seeds diverge more.',
            seedVarianceRandomize: 'Randomized channels',
            seedVarianceStrength: 'Variance strength',
            timestepShift: 'Shift',
            qwenImageShiftHelp: 'Auto derives the schedule shift from the image size; Lightning LoRAs expect 3.',
            timestepShiftHelp: 'Auto derives the schedule shift from the image size.',
            recentSeeds: 'Recent seeds',
            render: 'Render',
            resetToDefault: 'Reset to default',
            scheduler: 'Scheduler',
            seedSummary: '{{mode}} · {{seed}}',
            steps: 'Steps',
            useSeed: 'Use seed {{seed}}',
            useModelDefaultScheduler: 'Use model default scheduler',
          },
        },
      },
    },
  },
});

/** FLUX Fill defaults to guidance 30, above its track ceiling of 10. */
const fluxFillModel: MainModelConfig = {
  base: 'flux',
  default_settings: { cfg_scale: 1, guidance: 30 },
  format: 'diffusers',
  key: 'flux-fill',
  name: 'FLUX Fill',
  type: 'main',
};

/** `flux2_denoise.guidance` is `le=20`, which the capability table serves as `guidance_max`. */
const flux2Model: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2',
  name: 'FLUX.2 dev',
  type: 'main',
  variant: 'dev',
};

/** `ernie_image_denoise.guidance_scale` is `ge=1.0`, served as `guidance_min`. */
const ernieModel: MainModelConfig = {
  base: 'ernie-image',
  format: 'diffusers',
  key: 'ernie',
  name: 'ERNIE Image',
  type: 'main',
};

const ideogram4Model: MainModelConfig = {
  base: 'ideogram-4',
  format: 'diffusers',
  key: 'ideogram-4',
  name: 'Ideogram 4',
  type: 'main',
};

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' };
const mossImageModel: MainModelConfig = {
  base: 'moss-image',
  format: 'diffusers',
  key: 'moss-image',
  name: 'MOSS-Image',
  type: 'main',
};
const fluxDevModel: MainModelConfig = { base: 'flux', key: 'flux-dev', name: 'FLUX.1 dev', type: 'main' };
const zImageModel: MainModelConfig = {
  base: 'z-image',
  format: 'diffusers',
  key: 'z-image',
  name: 'Z-Image',
  type: 'main',
};
const qwenImageModel: MainModelConfig = {
  base: 'qwen-image',
  format: 'diffusers',
  key: 'qwen-image',
  name: 'Qwen Image',
  type: 'main',
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
/** The rendered section's draft, for changes that arrive from outside the field, as a recall does. */
let draftStore: ExternalStoreCore<GenerateSettings> | null = null;
/** The rendered section's immediate commits, which discrete edits use instead of the debounced `onCommit`. */
let onCommitImmediate: ReturnType<typeof vi.fn<(patch: Partial<GenerateSettings>) => void>> | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

/** `applyCommits` writes committed patches back to the draft, as the panel does, for fields that read them back. */
const render = async (model: GenerateModelConfig, settings: Partial<GenerateSettings> = {}, applyCommits = false) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const draft = createExternalStoreCore({ ...getDefaultGenerateSettings(model), ...settings });
  draftStore = draft;
  const applyCommit = (patch: Partial<GenerateSettings>) => {
    if (applyCommits) {
      draft.patchSnapshot(patch);
    }
  };
  const onCommit = vi.fn(applyCommit);
  const commitImmediate = vi.fn(applyCommit);
  onCommitImmediate = commitImmediate;

  await settle(() => {
    root?.render(
      // The seed field observes the dynamic prompts expansion query.
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <GenerateRenderSection
              draft={draft}
              selectedModel={model}
              onCommit={onCommit}
              onCommitImmediate={commitImmediate}
            />
          </I18nextProvider>
        </ChakraProvider>
      </QueryClientProvider>
    );
  });

  return onCommit;
};

const renderSeed = (overrides: Partial<GenerateSettings>) => render(sd1Model, { seed: 42, ...overrides });
const seedInput = () => host?.querySelector<HTMLInputElement>('input[aria-label="Seed"]') ?? null;
const modeTrigger = () => host?.querySelector<HTMLButtonElement>('button[aria-label^="Seed mode:"]') ?? null;
const menuItem = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')].find((item) =>
    item.textContent?.startsWith(label)
  ) ?? null;
const preview = () => host?.querySelector('[data-testid="seed-sequence-preview"]')?.textContent ?? null;

afterEach(async () => {
  harness.invocationSourceId = 'generate';
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  draftStore = null;
  onCommitImmediate = null;
});

describe('GenerateRenderSection seed field', () => {
  it('quiets the input in random mode but keeps the entered seed on show', async () => {
    await renderSeed({ seedMode: 'random' });

    expect(seedInput()?.disabled).toBe(true);
    expect(seedInput()?.value).toBe('42');
    expect(host?.querySelector<HTMLButtonElement>('button[aria-label="New seed"]')?.disabled).toBe(true);
    expect(modeTrigger()?.getAttribute('aria-label')).toBe('Seed mode: Random');
    expect(preview()).toBeNull();
  });

  it('commits a mode chosen from the menu without touching the seed', async () => {
    const onCommit = await renderSeed({ seedMode: 'random' });

    await settle(() => modeTrigger()?.click());
    const increment = menuItem('Increment');

    expect(increment?.getAttribute('aria-checked')).toBe('false');
    expect(menuItem('Random')?.getAttribute('aria-checked')).toBe('true');
    expect(increment?.textContent).toContain('Use successive seeds, increasing by 1.');

    await settle(() => increment?.click());

    expect(onCommit).toHaveBeenCalledWith({ seedMode: 'increment' });
  });

  it('is operable from the keyboard and returns focus to the trigger', async () => {
    const onCommit = await renderSeed({ seedMode: 'random' });
    const trigger = modeTrigger();

    await settle(() => trigger?.focus());
    await settle(() => trigger?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' })));
    // The menu itself takes focus and tracks the highlighted item by attribute.
    const menu = document.activeElement;

    expect(menu?.getAttribute('role')).toBe('menu');

    // Enter opens on the first mode; one step down lands on the second.
    await settle(() => menu?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));

    expect(document.querySelector('[role="menuitemradio"][data-highlighted]')?.getAttribute('data-value')).toBe(
      'fixed'
    );

    await settle(() => menu?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' })));

    expect(onCommit).toHaveBeenCalledWith({ seedMode: 'fixed' });
    expect(document.activeElement).toBe(trigger);
  });

  it('previews the seeds the next batch will run in a stepping mode', async () => {
    await renderSeed({ batchCount: 3, seedMode: 'increment' });

    expect(seedInput()?.disabled).toBe(false);
    expect(preview()).toBe('Next batch: 42 → 44');
    expect(document.getElementById(seedInput()?.getAttribute('aria-describedby') ?? '')?.textContent).toBe(
      'Next batch: 42 → 44'
    );
  });

  it('previews a single-seed batch without a range', async () => {
    await renderSeed({ batchCount: 1, seedMode: 'decrement' });

    expect(preview()).toBe('Next batch: 42');
  });

  it('pins a recent seed by switching to fixed', async () => {
    const onCommit = await renderSeed({ seedMode: 'increment' });

    await settle(() => host?.querySelector<HTMLButtonElement>('button[aria-label="Use seed 888"]')?.click());

    expect(onCommit).toHaveBeenCalledWith({ seed: 888, seedMode: 'fixed' });
  });

  it('names the active mode in the collapsed summary', async () => {
    await renderSeed({ seedMode: 'decrement' });

    expect(host?.textContent).toContain('Decrement · 42');
  });
});

describe('GenerateRenderSection denoising strength', () => {
  it('adds the canvas strength to the header badges and under guidance only when the canvas is the destination', async () => {
    await render(sd1Model);

    expect(host?.querySelector('[data-testid="strength-field"]')).toBeNull();
    expect(host?.querySelector('[data-testid="strength-badge"]')).toBeNull();

    await settle(() => root?.unmount());
    host?.remove();
    harness.invocationSourceId = 'canvas';
    await render(sd1Model);

    const header = host?.querySelector('[data-part="trigger"]');
    expect(header?.querySelector('[data-testid="strength-badge"]')).not.toBeNull();
    // In document order, the strength field is the control right after guidance.
    const controls = [...(host?.querySelectorAll('[data-scope="scrubber"], [data-testid="strength-field"]') ?? [])];
    const guidanceIndex = controls.indexOf(guidanceFrame('CFG')!);
    expect(guidanceIndex).toBeGreaterThan(-1);
    expect(controls[guidanceIndex + 1]?.getAttribute('data-testid')).toBe('strength-field');
  });

  it('adds no strength for a hosted provider, which edits the canvas composite at its own', async () => {
    harness.invocationSourceId = 'canvas';
    await render({
      base: 'external',
      capabilities: { modes: ['txt2img', 'img2img'] },
      format: 'external_api',
      key: 'gpt-image',
      name: 'GPT Image',
      provider_id: 'openai',
      type: 'external_image_generator',
    });

    expect(host?.querySelector('[data-testid="strength-field"]')).toBeNull();
    expect(host?.querySelector('[data-testid="strength-badge"]')).toBeNull();
  });
});

/** Resolve names through both aria-labelledby and aria-label. */
const sliderName = (slider: Element): string | null => {
  const labelId = slider.getAttribute('aria-labelledby');

  return (
    slider.getAttribute('aria-label') ??
    (labelId === null ? null : (host?.querySelector(`#${CSS.escape(labelId)}`)?.textContent ?? null))
  );
};
const slidersNamed = (label: string): Element[] =>
  [...(host?.querySelectorAll('[role="slider"]') ?? [])].filter((slider) => sliderName(slider) === label);
const guidanceSlider = (label = 'Guidance'): Element | undefined => slidersNamed(label)[0];
const guidanceFrame = (label = 'Guidance'): Element | null =>
  guidanceSlider(label)?.closest('[data-scope="scrubber"]') ?? null;
const guidanceValue = (label = 'Guidance'): string | undefined =>
  guidanceFrame(label)?.querySelector('button[data-part="value"]')?.textContent ?? undefined;

/** Opens the scrubber's editor on the shown value and closes it without a change, as a focus-out does. */
const openAndBlurEditor = async (label = 'Guidance') => {
  await settle(() => guidanceFrame(label)?.querySelector<HTMLButtonElement>('button[data-part="value"]')?.click());
  const editor = guidanceFrame(label)?.querySelector<HTMLInputElement>('input[data-part="value"]');

  if (!editor) {
    throw new Error(`The ${label} scrubber did not open its editor`);
  }

  await settle(() => editor.blur());
};

const pressOnSlider = (label: string, key: string) =>
  settle(() => guidanceSlider(label)?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key })));

describe('GenerateRenderSection guidance field', () => {
  // Seed capabilities so policy resolution is nonempty.
  seedArchitectureCapabilities();

  it('does not clamp a model default above the slider track when the editor closes', async () => {
    const onCommit = await render(fluxFillModel);

    expect(guidanceValue()).toBe('30');

    await openAndBlurEditor();

    // Assert committed values; controlled props can hide unintended clamping.
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('still holds the guidance track itself to its practical range', async () => {
    const onCommit = await render(fluxFillModel);

    // End targets the slider track, not the input ceiling.
    await pressOnSlider('Guidance', 'End');

    expect(onCommit).toHaveBeenCalledWith({ cfgScale: 10 });
  });

  it('clamps a typed guidance to the ceiling the architecture declares', async () => {
    // Exercise the FLUX Fill-to-FLUX.2 bound transition.
    const onCommit = await render(flux2Model, { cfgScale: 30 });

    await openAndBlurEditor();

    expect(onCommit).toHaveBeenCalledWith({ cfgScale: 20 });
  });

  it('starts the guidance track at the floor the architecture declares', async () => {
    // ERNIE guidance has a minimum of 1.
    const onCommit = await render(ernieModel, { cfgScale: 0.5 });

    await openAndBlurEditor('CFG');

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(onCommit).toHaveBeenCalledWith({ cfgScale: 1 });

    await pressOnSlider('CFG', 'Home');

    expect(onCommit).toHaveBeenCalledTimes(2);
    expect(onCommit).toHaveBeenLastCalledWith({ cfgScale: 1 });
  });

  it('holds the Ideogram 4 overrides to their own node bounds', async () => {
    // Ideogram override bounds come from numeric anyOf branches.
    await render(ideogram4Model, { ideogram4GuidanceScale: 5, ideogram4Mu: 1, ideogram4Steps: 48 });
    const ranges = (label: string) =>
      slidersNamed(label).map((slider) => [slider.getAttribute('aria-valuemin'), slider.getAttribute('aria-valuemax')]);

    // Address shared guidance before the override; both controls have the same accessible label.
    expect(ranges('Guidance')).toEqual([
      ['1', '10'],
      ['1', '20'],
    ]);
    expect(ranges('Steps')).toEqual([
      ['1', '100'],
      ['2', '100'],
    ]);
    expect(ranges('Mu')).toEqual([['-4', '4']]);

    // Off hands Mu to the sampler preset, and its description says so.
    const muInput = [...(host?.querySelectorAll<HTMLInputElement>('input[type="checkbox"]') ?? [])].find(
      (input) => document.getElementById(input.getAttribute('aria-labelledby') ?? '')?.textContent === 'Mu'
    );
    expect(document.getElementById(muInput?.getAttribute('aria-describedby') ?? '')?.textContent).toContain(
      'Set by the sampler preset unless overridden.'
    );
  });

  it('names the broken bound on the field, not only in the Invoke button tooltip', async () => {
    // Show local errors for persisted/recalled values that bypass selection clamps.
    await render(flux2Model, { cfgScale: 30 });

    expect(host?.querySelector('[role="alert"]')?.textContent).toBe('Guidance must be at most 20 for FLUX.2 dev.');
    expect(guidanceValue()).toBe('30');
    expect(guidanceFrame()?.hasAttribute('data-invalid')).toBe(true);
  });

  it('says nothing on the field while the value is inside the architecture bound', async () => {
    await render(flux2Model, { cfgScale: 7 });

    expect(host?.querySelector('[role="alert"]')).toBeNull();
  });

  it('drops the model-default mark the guidance track cannot place', async () => {
    // Exclude off-track default marks while retaining in-range marks.
    await render(fluxFillModel);
    const [stepsFrame, guidanceFrameElement] = [...(host?.querySelectorAll('[data-scope="scrubber"]') ?? [])];

    expect(stepsFrame?.querySelectorAll('[data-part="mark"]')).toHaveLength(1);
    expect(guidanceFrameElement?.querySelectorAll('[data-part="mark"]')).toHaveLength(0);
  });
});

describe('GenerateRenderSection before the capability table arrives', () => {
  // Use wide fallback bounds while generation is gated to avoid premature destructive clamps.
  it('keeps the guidance field permissive rather than guessing a bound', async () => {
    const onCommit = await render(fluxFillModel, { cfgScale: 30 });

    expect(guidanceValue('CFG')).toBe('30');
    expect(host?.querySelector('[role="alert"]')).toBeNull();

    await openAndBlurEditor('CFG');

    expect(onCommit).not.toHaveBeenCalled();
  });
});

describe('GenerateRenderSection model-family tuning', () => {
  seedArchitectureCapabilities();

  /** Select triggers are named by `aria-labelledby`, so resolve them through the field label. */
  const selectTrigger = (label: string): HTMLElement | null =>
    [...(host?.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="trigger"]') ?? [])].find((trigger) => {
      const labelId = trigger.getAttribute('aria-labelledby');

      return labelId !== null && host?.querySelector(`#${CSS.escape(labelId)}`)?.textContent === label;
    }) ?? null;
  const switchNamed = (label: string): HTMLElement | null =>
    [...(host?.querySelectorAll<HTMLElement>('[data-scope="switch"][data-part="root"]') ?? [])].find(
      (element) => element.textContent === label
    ) ?? null;

  it('offers DyPE only for FLUX.1, revealing scale and exponent under the manual preset', async () => {
    await render(sd1Model);
    expect(selectTrigger('DyPE')).toBeNull();

    await settle(() => root?.unmount());
    await render(fluxDevModel, { fluxDypePreset: 'auto' });
    expect(selectTrigger('DyPE')?.textContent).toContain('Auto (above 1536px)');
    expect(slidersNamed('DyPE scale (λs)')).toHaveLength(0);

    await settle(() => root?.unmount());
    await render(fluxDevModel, { fluxDypePreset: 'manual' });
    expect(slidersNamed('DyPE scale (λs)')[0]?.getAttribute('aria-valuemax')).toBe('8');
    expect(slidersNamed('DyPE exponent (λt)')).toHaveLength(1);
  });

  it('offers the shift override to Z-Image and Qwen Image, and seed variance to Z-Image only', async () => {
    const onCommit = await render(zImageModel);
    const override = switchNamed('Shift');

    expect(override).not.toBeNull();
    expect(slidersNamed('Shift')).toHaveLength(0);
    expect(switchNamed('Seed variance')).not.toBeNull();

    // Turning the override on starts at the Lightning shift rather than an empty field.
    await settle(() => override?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onCommit).toHaveBeenCalledWith({ zImageShift: 3 });

    await settle(() => root?.unmount());
    const onOverriddenCommit = await render(zImageModel, { zImageShift: 2.5 });
    expect(slidersNamed('Shift')[0]?.getAttribute('aria-valuenow')).toBe('2.5');
    // Turning the override off hands the shift back to the node, rather than leaving the last value in place.
    await settle(() => switchNamed('Shift')?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onOverriddenCommit).toHaveBeenCalledWith({ zImageShift: null });

    await settle(() => root?.unmount());
    const onQwenCommit = await render(qwenImageModel);
    expect(switchNamed('Seed variance')).toBeNull();
    await settle(() => switchNamed('Shift')?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onQwenCommit).toHaveBeenCalledWith({ qwenImageShift: 3 });

    await settle(() => root?.unmount());
    await render(sd1Model);
    expect(switchNamed('Shift')).toBeNull();
  });

  it('toggles the shift override from its label without handing focus to the slider it pairs with', async () => {
    const onCommit = await render(zImageModel, { zImageShift: 2.5 }, true);
    const label = () => switchNamed('Shift')?.querySelector<HTMLElement>('[data-part="label"]');
    const shiftInput = host?.querySelector<HTMLInputElement>('input[aria-label="Shift"]');

    expect(shiftInput).not.toBeNull();
    await act(() => userEvent.click(label()!));

    expect(onCommit).toHaveBeenCalledWith({ zImageShift: null });
    expect(document.activeElement).not.toBe(shiftInput);
    expect(slidersNamed('Shift')).toHaveLength(0);

    await act(() => userEvent.click(label()!));
    expect(onCommit).toHaveBeenLastCalledWith({ zImageShift: 3 });
  });

  it('keeps the shift override while a typed value passes outside its bounds', async () => {
    const onCommit = await render(zImageModel, { zImageShift: 3 });
    const input = () => host?.querySelector<HTMLInputElement>('input[aria-label="Shift"]');

    await settle(() => input()?.select());
    await act(() => userEvent.keyboard('0.5'));

    // "0" and "0." on the way to 0.5 are below the minimum, and committing them would read as auto.
    expect(onCommit.mock.calls.map(([patch]) => patch.zImageShift)).not.toContain(0);
    expect(onCommit).toHaveBeenLastCalledWith({ zImageShift: 0.5 });

    onCommit.mockClear();
    await settle(() => input()?.select());
    await act(() => userEvent.keyboard('12'));

    // "1" is in bounds; "12" is above the maximum and would read as auto too.
    expect(onCommit.mock.calls.map(([patch]) => patch.zImageShift)).toEqual([1]);
  });

  describe('MOSS-Image skip layers', () => {
    const input = () => host?.querySelector<HTMLInputElement>('input[aria-label="Skip layers (negative branch)"]');
    const alert = () => host?.querySelector('[role="alert"]')?.textContent ?? null;
    const typeOver = async (text: string) => {
      await settle(() => input()?.select());
      await act(() => userEvent.keyboard(text));
    };

    const stored = () => draftStore?.getSnapshot().mossImageUncondSkipLayers;

    it('commits a typed list only when the field loses focus, then shows it unique and ascending', async () => {
      const onCommit = await render(mossImageModel, {}, true);

      expect(input()?.value).toBe('22');
      expect(input()?.disabled).toBe(false);
      expect(host?.textContent).toContain('Default: 22. Empty skips none.');

      await typeOver('26, 22,24');

      // Typing commits nothing, so no prefix typed on the way reaches the settings.
      expect(onCommitImmediate).not.toHaveBeenCalled();
      expect(stored()).toEqual([22]);
      expect(input()?.value).toBe('26, 22,24');

      await settle(() => input()?.blur());
      expect(onCommitImmediate).toHaveBeenCalledExactlyOnceWith({ mossImageUncondSkipLayers: [22, 24, 26] });
      expect(input()?.value).toBe('22, 24, 26');

      // Clearing the field is a choice too: the negative branch then skips no block.
      await typeOver('{Backspace}');
      await settle(() => input()?.blur());
      expect(onCommitImmediate).toHaveBeenLastCalledWith({ mossImageUncondSkipLayers: [] });
      expect(input()?.getAttribute('aria-invalid')).toBeNull();
      expect(onCommit).not.toHaveBeenCalled();
    });

    it.each(['42', '22, 50', '22, x'])(
      'names %j while editing, and restores the list from before on blur',
      async (typed) => {
        await render(mossImageModel, { mossImageUncondSkipLayers: [24] }, true);

        await typeOver(typed);
        expect(input()?.getAttribute('aria-invalid')).toBe('true');
        expect(alert()).toBe('Enter comma-separated whole numbers from 0 to 41.');

        await settle(() => input()?.blur());
        expect(input()?.value).toBe('24');
        expect(alert()).toBeNull();
        expect(stored()).toEqual([24]);
        expect(onCommitImmediate).not.toHaveBeenCalled();
      }
    );

    it('commits a valid list when drafts flush before Invoke, without leaving the field', async () => {
      await render(mossImageModel, { mossImageUncondSkipLayers: [24] }, true);

      await typeOver('26, 22');
      await settle(() => flushWorkbenchDrafts());

      expect(document.activeElement).toBe(input());
      expect(stored()).toEqual([22, 26]);
      expect(input()?.value).toBe('22, 26');
    });

    it('restores the list from before when drafts flush over text that is not one', async () => {
      await render(mossImageModel, { mossImageUncondSkipLayers: [24] }, true);

      await typeOver('22, 50');
      expect(alert()).not.toBeNull();

      await settle(() => flushWorkbenchDrafts());
      expect(input()?.value).toBe('24');
      expect(alert()).toBeNull();
      expect(stored()).toEqual([24]);
      expect(onCommitImmediate).not.toHaveBeenCalled();
    });

    it('replaces typed text when the list changes from elsewhere', async () => {
      await render(mossImageModel, {}, true);

      await typeOver('22, x');
      // A recall or preset writes the draft without going through the field.
      await settle(() => draftStore?.patchSnapshot({ mossImageUncondSkipLayers: [24] }));

      expect(input()?.value).toBe('24');
      expect(alert()).toBeNull();

      // The stale text is gone for good: leaving the field commits nothing over the new list.
      await settle(() => input()?.blur());
      expect(stored()).toEqual([24]);
      expect(onCommitImmediate).not.toHaveBeenCalled();
    });

    it('resets to the default list from the field menu', async () => {
      await render(mossImageModel, { mossImageUncondSkipLayers: [26] }, true);
      const resetItem = () =>
        [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
          (item) => item.textContent === 'Reset to default'
        );

      await settle(() => input()?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true })));
      expect(resetItem()?.getAttribute('aria-disabled')).toBeNull();
      await act(() => userEvent.click(resetItem()!));

      expect(onCommitImmediate).toHaveBeenLastCalledWith({ mossImageUncondSkipLayers: [22] });
      expect(input()?.value).toBe('22');
    });

    it('rests disabled without CFG, and is offered to MOSS-Image only', async () => {
      await render(mossImageModel, { cfgScale: 1, mossImageUncondSkipLayers: [42] });

      expect(input()?.disabled).toBe(true);
      // The list has no effect without CFG, so it raises no error there.
      expect(alert()).toBeNull();
      expect(host?.textContent).toContain('Takes effect with CFG above 1, which runs the negative pass.');

      await settle(() => root?.unmount());
      await render(sd1Model);
      expect(input()).toBeNull();
    });
  });

  it('commits Z-Image seed variance to the Z-Image settings', async () => {
    const onCommit = await render(zImageModel);

    await settle(() => switchNamed('Seed variance')?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onCommit).toHaveBeenCalledWith({ zImageSeedVarianceEnabled: true });

    await settle(() => root?.unmount());
    const onEnabledCommit = await render(zImageModel, {
      zImageSeedVarianceEnabled: true,
      zImageSeedVarianceRandomizePercent: 50,
      zImageSeedVarianceStrength: 0.1,
    });
    await pressOnSlider('Variance strength', 'ArrowRight');
    await pressOnSlider('Randomized channels', 'ArrowRight');

    expect(onEnabledCommit).toHaveBeenCalledWith({ zImageSeedVarianceStrength: expect.closeTo(0.15) });
    expect(onEnabledCommit).toHaveBeenCalledWith({ zImageSeedVarianceRandomizePercent: 51 });
  });
});
