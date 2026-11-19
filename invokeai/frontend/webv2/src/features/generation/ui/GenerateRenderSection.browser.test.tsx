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
import { page, userEvent } from 'vitest/browser';

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
    CanvasDenoisingStrength: ({ children }: CanvasDenoisingStrengthProps) =>
      children({
        summary: <span data-testid="strength-badge">75%</span>,
        field: <div data-testid="strength-field" />,
      }),
    models: { getBaseLabel: (base: string) => ({ 'sd-1': 'SD 1.x', sdxl: 'SDXL' })[base] ?? base },
    project: { invocationSourceId: harness.invocationSourceId },
    sectionPreferences: { sectionsOpen: { render: true }, setSectionsOpen: vi.fn() },
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
            sectionMenu: {
              resetStepsAndGuidance: 'Reset steps and {{guidance}} to model defaults',
              resetStepsAndGuidanceBase: 'Reset steps and {{guidance}} to {{base}} defaults',
              resetStepsGuidanceAndScheduler: 'Reset steps, {{guidance}} and scheduler to model defaults',
              resetStepsGuidanceAndSchedulerBase: 'Reset steps, {{guidance}} and scheduler to {{base}} defaults',
              resetStepsGuidanceAndSchedulerMixed:
                'Reset steps, {{guidance}} and scheduler to model and {{base}} defaults',
            },
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

const flux2Model: MainModelConfig = {
  base: 'flux2',
  format: 'diffusers',
  key: 'flux2',
  name: 'FLUX.2 dev',
  type: 'main',
  variant: 'dev',
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
let host: HTMLDivElement | null = null;
let root: Root | null = null;
let draftStore: ExternalStoreCore<GenerateSettings> | null = null;
let onCommitImmediate: ReturnType<typeof vi.fn<(patch: Partial<GenerateSettings>) => void>> | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

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
const modeTrigger = () => host?.querySelector<HTMLButtonElement>('button[aria-label^="Seed mode:"]') ?? null;
const menuItem = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')].find((item) =>
    item.textContent?.startsWith(label)
  ) ?? null;
// Summary separators are CSS-drawn, so join the items as the section's copy action does.
const summaryText = () =>
  [...(host?.querySelectorAll('[data-section-summary] > *') ?? [])].map((item) => item.textContent?.trim()).join(' · ');

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
    const menu = document.activeElement;

    expect(menu?.getAttribute('role')).toBe('menu');

    await settle(() => menu?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));

    expect(document.querySelector('[role="menuitemradio"][data-highlighted]')?.getAttribute('data-value')).toBe(
      'fixed'
    );

    await settle(() => menu?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' })));

    expect(onCommit).toHaveBeenCalledWith({ seedMode: 'fixed' });
    expect(document.activeElement).toBe(trigger);
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
    const controls = [...(host?.querySelectorAll('[data-scope="scrubber"], [data-testid="strength-field"]') ?? [])];
    const guidanceIndex = controls.indexOf(guidanceFrame('CFG')!);
    expect(guidanceIndex).toBeGreaterThan(-1);
    expect(controls[guidanceIndex + 1]?.getAttribute('data-testid')).toBe('strength-field');
  });
});

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
const openAndBlurEditor = async (label = 'Guidance') => {
  await settle(() => guidanceFrame(label)?.querySelector<HTMLButtonElement>('button[data-part="value"]')?.click());
  const editor = guidanceFrame(label)?.querySelector<HTMLInputElement>('input[data-part="value"]');

  if (!editor) {
    throw new Error(`The ${label} scrubber did not open its editor`);
  }

  await settle(() => editor.blur());
};

describe('GenerateRenderSection guidance field', () => {
  seedArchitectureCapabilities();

  it('clamps a typed guidance to the ceiling the architecture declares', async () => {
    const onCommit = await render(flux2Model, { cfgScale: 30 });

    await openAndBlurEditor();

    expect(onCommit).toHaveBeenCalledWith({ cfgScale: 20 });
  });

  it('summarizes Ideogram 4 sampling and offers only the preset overrides', async () => {
    await render(ideogram4Model, { cfgScale: 3, ideogram4SamplerPreset: 'V4_TURBO_12', steps: 99 });
    const ranges = (label: string) =>
      slidersNamed(label).map((slider) => [slider.getAttribute('aria-valuemin'), slider.getAttribute('aria-valuemax')]);

    expect(summaryText()).toContain('12 · Guidance 7');
    expect(ranges('Guidance')).toEqual([]);
    expect(ranges('Steps')).toEqual([]);

    await settle(() => draftStore?.patchSnapshot({ ideogram4GuidanceScale: 5, ideogram4Mu: 1, ideogram4Steps: 18 }));

    expect(summaryText()).toContain('18 · Guidance 5');
    expect(ranges('Guidance')).toEqual([['1', '20']]);
    expect(ranges('Steps')).toEqual([['2', '100']]);
    expect(ranges('Mu')).toEqual([['-4', '4']]);

    // Off hands Mu to the sampler preset, and its description says so.
    const muInput = [...(host?.querySelectorAll<HTMLInputElement>('input[type="checkbox"]') ?? [])].find(
      (input) => document.getElementById(input.getAttribute('aria-labelledby') ?? '')?.textContent === 'Mu'
    );
    expect(document.getElementById(muInput?.getAttribute('aria-describedby') ?? '')?.textContent).toContain(
      'Set by the sampler preset unless overridden.'
    );
  });
});

describe('GenerateRenderSection model-family tuning', () => {
  seedArchitectureCapabilities();

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

  describe('MOSS-Image skip layers', () => {
    const input = () => host?.querySelector<HTMLInputElement>('input[aria-label="Skip layers (negative branch)"]');
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

      expect(onCommitImmediate).not.toHaveBeenCalled();
      expect(stored()).toEqual([22]);
      expect(input()?.value).toBe('26, 22,24');

      await settle(() => input()?.blur());
      expect(onCommitImmediate).toHaveBeenCalledExactlyOnceWith({ mossImageUncondSkipLayers: [22, 24, 26] });
      expect(input()?.value).toBe('22, 24, 26');

      await typeOver('{Backspace}');
      await settle(() => input()?.blur());
      expect(onCommitImmediate).toHaveBeenLastCalledWith({ mossImageUncondSkipLayers: [] });
      expect(input()?.getAttribute('aria-invalid')).toBeNull();
      expect(onCommit).not.toHaveBeenCalled();
    });

    it('commits a valid list when drafts flush before Invoke, without leaving the field', async () => {
      await render(mossImageModel, { mossImageUncondSkipLayers: [24] }, true);

      await typeOver('26, 22');
      await settle(() => flushWorkbenchDrafts());

      expect(document.activeElement).toBe(input());
      expect(stored()).toEqual([22, 26]);
      expect(input()?.value).toBe('22, 26');
    });
  });
});

describe('GenerateRenderSection title defaults', () => {
  seedArchitectureCapabilities();

  const title = () => host!.querySelector<HTMLElement>('.generate-section > [data-part="trigger"]')!;
  const isMarked = () => title().querySelector('[data-section-changed]') !== null;
  const otherScheduler = (model: GenerateModelConfig) =>
    getDefaultGenerateSettings(model).scheduler === 'ddim' ? 'euler' : 'ddim';
  const openMenu = () =>
    act(() => {
      title().dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });

  it('marks steps, CFG and scheduler off the model defaults, and resets those three only', async () => {
    const defaults = getDefaultGenerateSettings(sd1Model);
    const onCommit = await render(sd1Model, { scheduler: otherScheduler(sd1Model), seed: 42, seedMode: 'fixed' }, true);

    expect(host?.textContent).toContain('Scheduler');
    expect(isMarked()).toBe(true);
    await openMenu();
    await act(() =>
      userEvent.click(page.getByRole('menuitem', { name: 'Reset steps, CFG and scheduler to SD 1.x defaults' }))
    );

    expect(onCommit).toHaveBeenLastCalledWith({
      cfgScale: defaults.cfgScale,
      scheduler: defaults.scheduler,
      steps: defaults.steps,
    });
    expect(draftStore?.getSnapshot()).toMatchObject({ seed: 42, seedMode: 'fixed' });
    expect(isMarked()).toBe(false);
  });
});
