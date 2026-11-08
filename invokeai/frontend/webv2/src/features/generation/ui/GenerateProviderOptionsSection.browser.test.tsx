/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateModelConfig, GenerateSettings, MainModelConfig } from '@features/generation/core/types';

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

import { GenerateProviderOptionsSection } from './GenerateProviderOptionsSection';

vi.mock('./GenerationUiContext', () => ({
  useGenerationUi: () => ({
    project: { invocationSourceId: 'generate' },
    sectionPreferences: { sectionsOpen: { providerOptions: true }, setSectionOpen: vi.fn() },
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
        common: { scrubber: { editValue: 'Edit {{label}}' } },
        widgets: {
          generate: {
            overridesCount_one: '{{count}} override',
            overridesCount_other: '{{count}} overrides',
            providerOptions: {
              geminiTemperature: 'Temperature',
              geminiThinkingLevel: 'Thinking level',
              geminiThinkingLevelOptions: { high: 'High', minimal: 'Minimal' },
              openaiBackground: 'Background',
              openaiBackgroundOptions: { auto: 'Auto', opaque: 'Opaque', transparent: 'Transparent' },
              openaiInputFidelity: 'Input fidelity',
              openaiInputFidelityOptions: { high: 'High', low: 'Low' },
              openaiQuality: 'Quality',
              openaiQualityOptions: { auto: 'Auto', high: 'High', low: 'Low', medium: 'Medium' },
              providerDefault: 'Provider default',
              seedreamOptimizePrompt: 'Optimize prompt',
              seedreamWatermark: 'Watermark',
              title: 'Provider options',
            },
          },
        },
      },
    },
  },
});

const external = (
  providerId: string,
  capabilities: Record<string, unknown>,
  providerModelId = providerId === 'openai' ? 'gpt-image-1' : `${providerId}-model`
): GenerateModelConfig =>
  ({
    base: 'external',
    capabilities,
    format: 'external_api',
    key: `${providerId}-model`,
    name: providerId,
    provider_id: providerId,
    provider_model_id: providerModelId,
    type: 'external_image_generator',
  }) as GenerateModelConfig;
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

const render = async (model: GenerateModelConfig, settings: Partial<GenerateSettings> = {}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const draft = createExternalStoreCore({ ...getDefaultGenerateSettings(model), ...settings });
  // Commits land in the draft as the form's would, so the section re-renders from what it committed.
  const onCommit = vi.fn((patch: Partial<GenerateSettings>) => draft.setSnapshot({ ...draft.getSnapshot(), ...patch }));

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateProviderOptionsSection draft={draft} selectedModel={model} onCommit={onCommit} />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return onCommit;
};

const selectTriggers = () => [
  ...(host?.querySelectorAll<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]') ?? []),
];
const selectNames = () =>
  selectTriggers().map((trigger) =>
    (trigger.getAttribute('aria-labelledby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent?.trim() ?? '')
      .join(' ')
      .trim()
  );
const switches = () => [...(host?.querySelectorAll<HTMLInputElement>('input[type="checkbox"]') ?? [])];
const option = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((item) => item.textContent === label) ?? null;

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateProviderOptionsSection', () => {
  it('renders nothing for local models and providers without options', async () => {
    await render(sd1Model);
    expect(host?.childElementCount).toBe(0);

    await settle(() => root?.unmount());
    await render(external('alibabacloud', { modes: ['txt2img'] }));
    expect(host?.childElementCount).toBe(0);

    // OpenAI sends its options to GPT Image models only.
    await settle(() => root?.unmount());
    await render(external('openai', { modes: ['txt2img'] }, 'dall-e-3'));
    expect(host?.childElementCount).toBe(0);
  });

  it('offers OpenAI quality and background, adding input fidelity only for image inputs', async () => {
    await render(external('openai', { modes: ['txt2img'] }));
    expect(selectNames()).toEqual(['Quality', 'Background']);

    await settle(() => root?.unmount());
    const onCommit = await render(external('openai', { modes: ['txt2img', 'img2img'] }));
    expect(selectNames()).toEqual(['Quality', 'Background', 'Input fidelity']);

    await settle(() => selectTriggers()[1]?.click());
    await settle(() => option('Transparent')?.click());
    expect(onCommit).toHaveBeenCalledWith({ openaiBackground: 'transparent' });

    await settle(() => selectTriggers()[2]?.click());
    await settle(() => option('High')?.click());
    expect(onCommit).toHaveBeenCalledWith({ openaiInputFidelity: 'high' });
  });

  it('lets Gemini temperature be overridden and thinking level return to the provider default', async () => {
    const onCommit = await render(external('gemini', { modes: ['txt2img'] }), { geminiThinkingLevel: 'high' });

    expect(selectNames()).toEqual(['Thinking level']);
    // Temperature starts at the provider default: no slider until overridden.
    expect(host?.querySelector('[role="slider"]')).toBeNull();

    await settle(() => switches()[0]?.click());
    expect(onCommit).toHaveBeenCalledWith({ geminiTemperature: 1 });
    expect(host?.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('1');

    await settle(() => selectTriggers()[0]?.click());
    await settle(() => option('Provider default')?.click());
    expect(onCommit).toHaveBeenCalledWith({ geminiThinkingLevel: null });
  });

  it('toggles the Seedream flags and counts them as overrides', async () => {
    const onCommit = await render(external('seedream', { modes: ['txt2img'] }), { seedreamWatermark: true });

    expect(host?.textContent).toContain('1 override');
    const [watermark, optimizePrompt] = switches();

    expect(watermark?.checked).toBe(true);
    expect(optimizePrompt?.checked).toBe(false);

    // Each switch is named by, and toggled from, its own visible label.
    const labelOf = (input: HTMLInputElement | undefined) =>
      document.getElementById(input?.getAttribute('aria-labelledby') ?? '');

    expect(labelOf(watermark)?.textContent).toBe('Watermark');
    expect(labelOf(optimizePrompt)?.textContent).toBe('Optimize prompt');

    await act(() => userEvent.click(labelOf(optimizePrompt)!));
    expect(onCommit).toHaveBeenCalledWith({ seedreamOptimizePrompt: true });
    expect(onCommit).not.toHaveBeenCalledWith({ seedreamWatermark: false });
  });
});
