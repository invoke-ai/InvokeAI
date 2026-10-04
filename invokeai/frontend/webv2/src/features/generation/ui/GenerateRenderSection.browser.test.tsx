/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateRenderSection } from './GenerateRenderSection';

const seedHistory = [
  { seed: 777, thumbnailUrl: null },
  { seed: 888, thumbnailUrl: null },
];

vi.mock('./GenerationUiContext', () => ({
  useGenerationUi: () => ({
    queueInsights: { secondsPerRun: null, seedHistory },
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
          scrubber: { editValue: 'Edit {{label}}' },
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
            ideogram4MuHelp: 'Timestep shift. Higher values spend more of the schedule at high noise.',
            ideogram4PresetDerived: 'Set by the sampler preset unless overridden.',
            ideogram4SamplerPreset: 'Sampler preset',
            ideogram4Steps: 'Steps',
            override: 'Override',
            recentSeeds: 'Recent seeds',
            render: 'Render',
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

const render = async (model: MainModelConfig, settings: Partial<GenerateSettings> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onCommit = vi.fn();

  await settle(() => {
    root?.render(
      // The seed field observes the dynamic prompts expansion query.
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <GenerateRenderSection
              selectedModel={model}
              settings={{ ...getDefaultGenerateSettings(model), ...settings }}
              onCommit={onCommit}
              onCommitImmediate={vi.fn()}
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
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
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
