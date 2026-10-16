/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GenerationModelCatalogItem } from '@features/generation/contracts';
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

import type { GenerationModelSelectProps } from './GenerationUiContext';

import { GenerateRefinerSection } from './GenerateRefinerSection';

const sdxlModel: MainModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
const fluxModel: MainModelConfig = { base: 'flux', key: 'flux', name: 'FLUX.1 dev', type: 'main' };
const refinerModel: MainModelConfig = { base: 'sdxl-refiner', key: 'refiner', name: 'SDXL Refiner', type: 'main' };
const CATALOG: readonly GenerationModelCatalogItem[] = [sdxlModel, refinerModel];

vi.mock('./GenerationUiContext', () => ({
  /** The picker reduced to one button per catalog model that passes the section's filter. */
  GenerationModelSelect: ({ filter, onChange, scopeLabel, value }: GenerationModelSelectProps) => (
    <div data-testid="refiner-picker" data-scope-label={scopeLabel ?? ''} data-value={value ?? ''}>
      {CATALOG.filter((model) => filter?.(model) ?? true).map((model) => (
        <button key={model.key} type="button" onClick={() => onChange(model)}>
          {model.name}
        </button>
      ))}
    </div>
  ),
  useGenerationUi: () => ({
    sectionPreferences: { sectionsOpen: { refiner: true }, setSectionOpen: vi.fn() },
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
        commandPalette: { settings: { off: 'Off' } },
        widgets: {
          generate: {
            refiner: 'Refiner',
            refinerCfgScale: 'Refiner CFG',
            refinerModel: 'Refiner model',
            refinerModelsScope: 'SDXL Refiner models',
            refinerNegativeAestheticScore: 'Negative aesthetic score',
            refinerNone: 'None',
            refinerPositiveAestheticScore: 'Positive aesthetic score',
            refinerScheduler: 'Refiner scheduler',
            refinerStart: 'Refiner start',
            refinerSteps: 'Refiner steps',
          },
        },
      },
    },
  },
});

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

const render = async (model: MainModelConfig, options: Partial<GenerateSettings> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const settings = { ...getDefaultGenerateSettings(model), ...options };
  const draft = createExternalStoreCore(settings);
  const onCommit = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));
  const onCommitImmediate = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateRefinerSection
            draft={draft}
            selectedModel={model}
            onCommit={onCommit}
            onCommitImmediate={onCommitImmediate}
          />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return { onCommit, onCommitImmediate };
};

/** Sliders are named by `aria-labelledby`, so resolve them through their label element. */
const sliderByLabel = (label: string): HTMLElement | null =>
  [...(host?.querySelectorAll<HTMLElement>('[role="slider"]') ?? [])].find((control) => {
    const labelId = control.getAttribute('aria-labelledby');

    return labelId !== null && host?.querySelector(`#${CSS.escape(labelId)}`)?.textContent === label;
  }) ?? null;

const schedulerInput = (): HTMLInputElement | null =>
  host?.querySelector<HTMLInputElement>('input[aria-label="Refiner scheduler"]') ?? null;

const pickerButton = (name: string): HTMLButtonElement | null =>
  [...(host?.querySelectorAll<HTMLButtonElement>('[data-testid="refiner-picker"] button') ?? [])].find(
    (button) => button.textContent === name
  ) ?? null;

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateRefinerSection', () => {
  it('renders for SDXL only', async () => {
    await render(fluxModel);

    expect(host?.textContent).toBe('');

    await settle(() => root?.unmount());
    host?.remove();

    await render(sdxlModel);

    expect(host?.textContent).toContain('Refiner');
    expect(host?.querySelector('[data-testid="refiner-picker"]')).not.toBeNull();
  });

  it('offers only refiner models and keeps the tuning disabled until one is picked', async () => {
    const { onCommitImmediate } = await render(sdxlModel);

    expect(pickerButton('SDXL')).toBeNull();
    expect(pickerButton('SDXL Refiner')).not.toBeNull();
    expect(sliderByLabel('Refiner start')?.getAttribute('aria-disabled')).toBe('true');
    expect(sliderByLabel('Refiner steps')?.getAttribute('aria-disabled')).toBe('true');
    expect(sliderByLabel('Refiner CFG')?.getAttribute('aria-disabled')).toBe('true');
    expect(schedulerInput()?.disabled).toBe(true);

    await settle(() => pickerButton('SDXL Refiner')?.click());

    expect(onCommitImmediate).toHaveBeenCalledWith({ refinerModel: refinerModel });
  });

  it('enables the tuning once a refiner is selected and shows it in the header', async () => {
    await render(sdxlModel, { refinerModel });

    expect(host?.textContent).toContain('SDXL Refiner');
    expect(host?.querySelector('[data-testid="refiner-picker"]')?.getAttribute('data-value')).toBe('refiner');
    expect(sliderByLabel('Refiner start')?.getAttribute('aria-disabled')).not.toBe('true');
    expect(sliderByLabel('Refiner CFG')?.getAttribute('aria-disabled')).not.toBe('true');
    expect(schedulerInput()?.disabled).toBe(false);
    expect(sliderByLabel('Positive aesthetic score')?.getAttribute('aria-valuenow')).toBe('6');
    expect(sliderByLabel('Negative aesthetic score')?.getAttribute('aria-valuenow')).toBe('2.5');
  });

  it('shows Off at start 1, disables tuning, and lets the start control reactivate the refiner', async () => {
    const { onCommit } = await render(sdxlModel, { refinerModel, refinerStart: 1 });

    expect(host?.querySelector('button[data-scope="collapsible"]')?.textContent).toContain('Off');
    expect(sliderByLabel('Refiner start')?.getAttribute('aria-disabled')).not.toBe('true');
    expect(pickerButton('SDXL Refiner')?.disabled).toBe(false);
    expect(sliderByLabel('Refiner steps')?.getAttribute('aria-disabled')).toBe('true');
    expect(sliderByLabel('Refiner CFG')?.getAttribute('aria-disabled')).toBe('true');
    expect(sliderByLabel('Positive aesthetic score')?.getAttribute('aria-disabled')).toBe('true');
    expect(sliderByLabel('Negative aesthetic score')?.getAttribute('aria-disabled')).toBe('true');
    expect(schedulerInput()?.disabled).toBe(true);

    await settle(() => {
      const start = sliderByLabel('Refiner start');
      start?.focus();
      start?.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
      start?.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowLeft', bubbles: true }));
    });

    expect(onCommit).toHaveBeenCalledWith({ refinerStart: 0.99 });
    expect(sliderByLabel('Refiner steps')?.getAttribute('aria-disabled')).not.toBe('true');
    expect(schedulerInput()?.disabled).toBe(false);
    expect(host?.querySelector('button[data-scope="collapsible"]')?.textContent).toContain('SDXL Refiner');
  });

  it('leaves the empty and loading states to the picker, scoped to refiner models', async () => {
    await render(sdxlModel);

    expect(host?.querySelector('[data-testid="refiner-picker"]')?.getAttribute('data-scope-label')).toBe(
      'SDXL Refiner models'
    );
    // No section-level help text duplicates the picker's own empty message or flashes before the catalog loads.
    expect(host?.textContent).not.toContain('installed');
  });
});
