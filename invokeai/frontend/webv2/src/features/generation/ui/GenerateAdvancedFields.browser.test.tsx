/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateAdvancedFields } from './GenerateAdvancedFields';

vi.mock('./GenerationUiContext', () => ({
  GenerationModelSelect: () => null,
  useGenerationUi: () => ({
    sectionPreferences: { sectionsOpen: { advanced: true }, setSectionOpen: vi.fn() },
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
        widgets: {
          generate: {
            advanced: 'Advanced',
            cfgRescale: 'CFG rescale',
            clipSkip: 'CLIP skip',
            hiDiffusion: 'HiDiffusion',
            hiDiffusionRauNet: 'HiDiffusion: RAU-Net',
            hiDiffusionT1Ratio: 'HiDiffusion: T1 Ratio',
            hiDiffusionT2Ratio: 'HiDiffusion: T2 Ratio',
            hiDiffusionWindowAttn: 'HiDiffusion: Window Attention',
            seamlessTiling: 'Seamless tiling',
            tileX: 'Tile X',
            tileY: 'Tile Y',
            useModelDefault: 'Use model default',
            useModelDefaultCfgRescale: 'Use model default CFG rescale',
            useModelDefaultVae: 'Use bundled VAE',
            useModelDefaultVaePrecision: 'Use model default precision',
            usingBundledVae: 'Using bundled VAE',
            vae: 'VAE',
            vaePrecision: 'VAE precision',
            xAxis: 'X axis',
            yAxis: 'Y axis',
          },
        },
      },
    },
  },
});

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' };
const sd2Model: MainModelConfig = { base: 'sd-2', key: 'sd2', name: 'SD 2', type: 'main' };
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

const render = async (model: MainModelConfig, hiDiffusionEnabled: boolean) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const settings = { ...getDefaultGenerateSettings(model), hiDiffusionEnabled };
  const onCommit = vi.fn();

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateAdvancedFields
            selectedModel={model}
            settings={settings}
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
    expect(host?.querySelector<HTMLInputElement>('input[aria-label="HiDiffusion: T1 Ratio"]')?.disabled).toBe(true);

    await settle(() => main?.querySelector<HTMLElement>('[data-part="control"]')?.click());
    expect(onCommit).toHaveBeenCalledWith({ hiDiffusionEnabled: true });
  });

  it('omits HiDiffusion controls for SD2', async () => {
    await render(sd2Model, true);

    expect(switchByLabel('HiDiffusion')).toBeNull();
    expect(host?.textContent).not.toContain('HiDiffusion: T1 Ratio');
  });
});
