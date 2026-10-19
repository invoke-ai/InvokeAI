/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ExternalSizeSelect } from './ExternalSizeSelect';

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
            providerSize: 'Provider size',
            providerSizeCustom: 'Custom',
            providerSizeCanvasCropHelp: 'Canvas generates {{width}}×{{height}}, cropped to the frame; edits resized.',
            providerSizeCanvasCropTextHelp: 'Canvas generates {{width}}×{{height}}, cropped to the frame.',
            providerSizeCanvasResizeHelp: 'Canvas generates {{width}}×{{height}}, resized to the frame.',
            providerSizeHelp: 'Sizes this provider renders natively.',
          },
        },
      },
    },
  },
});

const external = (capabilities: Record<string, unknown>): GenerateModelConfig =>
  ({
    base: 'external',
    capabilities,
    format: 'external_api',
    key: 'external',
    name: 'External',
    provider_id: 'gemini',
    type: 'external_image_generator',
  }) as GenerateModelConfig;

const presetModel = external({
  modes: ['txt2img'],
  resolution_presets: [
    { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 },
    { aspect_ratio: '16:9', height: 1152, image_size: '2K', label: '16:9 (2K)', width: 2048 },
  ],
});
const ratioModel = external({
  aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 }, '7:4': { height: 1024, width: 1792 } },
  modes: ['txt2img'],
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

const render = async (
  model: GenerateModelConfig,
  settings: { externalImageSize: string | null; height: number; width: number },
  isCanvasFrame = false
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onCommit = vi.fn();

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <ExternalSizeSelect isCanvasFrame={isCanvasFrame} model={model} settings={settings} onCommit={onCommit} />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return onCommit;
};

const trigger = () => host?.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]') ?? null;
const option = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((item) => item.textContent === label) ?? null;

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ExternalSizeSelect', () => {
  it('renders nothing for providers without a size table', async () => {
    await render(external({ modes: ['txt2img'] }), { externalImageSize: null, height: 1024, width: 1024 });

    expect(host?.childElementCount).toBe(0);
  });

  it('applies a named preset as a locked size with its image size', async () => {
    const onCommit = await render(presetModel, { externalImageSize: '1K', height: 1024, width: 1024 });

    expect(trigger()?.textContent).toContain('1:1 (1K)');
    await settle(() => trigger()?.click());
    await settle(() => option('16:9 (2K)')?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2048 / 1152,
      externalImageSize: '2K',
      height: 1152,
      width: 2048,
    });
  });

  it('recognizes dimensions typed to a provider preset without a stored name', async () => {
    await render(presetModel, { externalImageSize: null, height: 1152, width: 2048 });

    expect(trigger()?.textContent).toContain('16:9 (2K)');
  });

  it('shows a hand-typed size as custom', async () => {
    await render(presetModel, { externalImageSize: '1K', height: 1000, width: 1024 });

    expect(trigger()?.textContent).toContain('Custom');
    expect(host?.textContent).toContain('Sizes this provider renders natively.');
  });

  it('tells a Canvas frame off the size table which size is generated and how it is fitted', async () => {
    // A text-only model is never edited, so the hint leaves edits out.
    await render(ratioModel, { externalImageSize: null, height: 512, width: 640 }, true);
    expect(host?.textContent).toContain('Canvas generates 1024×1024, cropped to the frame.');

    await settle(() => root?.unmount());
    host?.remove();
    await render(
      external({
        aspect_ratio_sizes: { '1:1': { height: 1024, width: 1024 }, '7:4': { height: 1024, width: 1792 } },
        modes: ['txt2img', 'img2img'],
      }),
      { externalImageSize: null, height: 512, width: 640 },
      true
    );
    expect(host?.textContent).toContain('Canvas generates 1024×1024, cropped to the frame; edits resized.');

    await settle(() => root?.unmount());
    host?.remove();
    await render(ratioModel, { externalImageSize: null, height: 512, width: 896 }, true);
    expect(host?.textContent).toContain('Canvas generates 1792×1024, resized to the frame.');
  });

  it('applies a fixed ratio size, falling back to a free ratio id for shapes outside the preset list', async () => {
    const onCommit = await render(ratioModel, { externalImageSize: null, height: 1024, width: 1024 });

    expect(trigger()?.textContent).toContain('1:1 (1024×1024)');
    await settle(() => trigger()?.click());
    await settle(() => option('7:4 (1792×1024)')?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 1792 / 1024,
      externalImageSize: null,
      height: 1024,
      width: 1792,
    });
  });
});
