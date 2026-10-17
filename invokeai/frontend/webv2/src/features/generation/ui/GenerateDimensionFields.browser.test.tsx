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

import type { CanvasRenderSizeProps } from './GenerationUiContext';

import { GenerateDimensionFields } from './GenerateDimensionFields';

const harness = vi.hoisted(() => ({ invocationSourceId: 'generate' }));

vi.mock('./GenerationUiContext', () => ({
  useGenerationQueueInsights: (select: (insights: unknown) => unknown) =>
    select({ secondsPerRun: null, seedHistory: [] }),
  useGenerationUi: () => ({
    // Stands in for the canvas's render-size slot, which lays out the Size section around its controls.
    CanvasRenderSize: ({ children, frame }: CanvasRenderSizeProps) =>
      children({
        controls: <div data-testid="render-size-controls" />,
        size: { height: frame.height * 2, width: frame.width * 2 },
      }),
    project: { invocationSourceId: harness.invocationSourceId },
    sectionPreferences: { sectionsOpen: { dimensions: true }, setSectionOpen: vi.fn() },
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
            aspectRatio: 'Aspect ratio',
            height: 'Height',
            lockAspectRatio: 'Lock aspect ratio',
            megapixelsValue: '{{value}} MP',
            size: 'Size',
            swapWidthAndHeight: 'Swap width and height',
            unlockAspectRatio: 'Unlock aspect ratio',
            width: 'Width',
          },
        },
      },
    },
  },
});

const sd1Model: MainModelConfig = { base: 'sd-1', key: 'sd1', name: 'SD 1.5', type: 'main' };
const gptImage: GenerateModelConfig = {
  base: 'external',
  capabilities: { modes: ['txt2img', 'img2img'] },
  format: 'external_api',
  key: 'gpt-image',
  name: 'GPT Image',
  provider_id: 'openai',
  type: 'external_image_generator',
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

const render = async (settings: Partial<GenerateSettings>, model: GenerateModelConfig = sd1Model) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onCommit = vi.fn();

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateDimensionFields
            draft={createExternalStoreCore({ ...getDefaultGenerateSettings(model), ...settings })}
            projectId="project"
            selectedModel={model}
            onCommit={onCommit}
          />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return onCommit;
};

const lockButton = () =>
  host?.querySelector<HTMLButtonElement>(
    'button[aria-label="Lock aspect ratio"], button[aria-label="Unlock aspect ratio"]'
  ) ?? null;
const bracket = () => host?.querySelector('[data-part="bracket"]')?.parentElement ?? null;
const presetTrigger = () => host?.querySelector<HTMLButtonElement>('[role="combobox"]') ?? null;

afterEach(async () => {
  harness.invocationSourceId = 'generate';
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateDimensionFields canvas render size', () => {
  it('lays out the render-size controls and the resized badge for a local model on the canvas', async () => {
    harness.invocationSourceId = 'canvas';
    await render({ height: 512, width: 512 });

    expect(host?.querySelector('[data-testid="render-size-controls"]')).not.toBeNull();
    expect(host?.textContent).toContain('512x512 → 1024x1024');
  });

  it('shows a hosted provider only the frame, which is the size the provider is asked for', async () => {
    harness.invocationSourceId = 'canvas';
    await render({ height: 512, width: 512 }, gptImage);

    expect(host?.querySelector('[data-testid="render-size-controls"]')).toBeNull();
    expect(host?.textContent).not.toContain('→');
    expect(host?.textContent).toContain('512x512');
  });
});

describe('GenerateDimensionFields aspect ratio', () => {
  it('reshapes the size to a chosen preset and locks it', async () => {
    const onCommit = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 1024, width: 1024 });

    expect(lockButton()?.getAttribute('aria-label')).toBe('Lock aspect ratio');
    expect(bracket()?.hasAttribute('data-locked')).toBe(false);

    await settle(() => presetTrigger()?.click());
    await settle(() => document.querySelector<HTMLElement>('[role="option"][data-value="16:9"]')?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1368,
    });
  });

  it('shows the lock engaged while a preset is selected and unlocking returns to Free', async () => {
    // A preset implies ratio lock even when the stored flag is false.
    const onCommit = await render({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: false,
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1368,
    });

    expect(presetTrigger()?.textContent).toContain('16:9');
    expect(lockButton()?.getAttribute('aria-label')).toBe('Unlock aspect ratio');
    expect(bracket()?.hasAttribute('data-locked')).toBe(true);

    await settle(() => lockButton()?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 1368 / 768,
      height: 768,
      width: 1368,
    });
  });

  it('locking in Free captures the current ratio', async () => {
    const onCommit = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 800, width: 1024 });

    await settle(() => lockButton()?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 1.28,
      height: 800,
      width: 1024,
    });
  });

  it('locking names the preset the current size already matches', async () => {
    // The same gesture in the canvas geometry form shows "4:3" for these dimensions.
    const onCommit = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 768, width: 1024 });

    await settle(() => lockButton()?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: '4:3',
      aspectRatioIsLocked: true,
      aspectRatioValue: 4 / 3,
      height: 768,
      width: 1024,
    });
  });

  it('swaps width and height and mirrors the preset', async () => {
    const onCommit = await render({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1368,
    });

    await settle(() => host?.querySelector<HTMLButtonElement>('button[aria-label="Swap width and height"]')?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: '9:16',
      aspectRatioValue: expect.closeTo(9 / 16, 10),
      height: 1368,
      width: 768,
    });
  });
});
