/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateModelConfig, GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/models/core/architectureCapabilities.testing';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { CanvasRenderSizeProps } from './GenerationUiContext';

import { GenerateDimensionFields } from './GenerateDimensionFields';

const harness = vi.hoisted(() => ({
  aspectRatioPresets: [] as { height: number; id: string; label: string; width: number }[],
  invocationSourceId: 'generate',
}));

vi.mock('./GenerationUiContext', () => ({
  useGenerationQueueInsights: (select: (insights: unknown) => unknown) =>
    select({ secondsPerRun: null, seedHistory: [] }),
  useGenerationUi: () => ({
    CanvasRenderSize: ({ children, frame }: CanvasRenderSizeProps) =>
      children({
        controls: <div data-testid="render-size-controls" />,
        size: { height: frame.height * 2, width: frame.width * 2 },
      }),
    aspectRatioPresets: {
      isFull: false,
      presets: harness.aspectRatioPresets,
      remove: vi.fn(),
      save: vi.fn(),
      update: vi.fn(),
    },
    models: { getBaseLabel: (base: string) => ({ 'sd-1': 'SD 1.x', sdxl: 'SDXL' })[base] ?? base },
    project: { invocationSourceId: harness.invocationSourceId },
    sectionPreferences: { sectionsOpen: { dimensions: true }, setSectionsOpen: vi.fn() },
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
        common: { scrubber: { editValue: 'Edit {{label}}: {{value}}' } },
        widgets: {
          generate: {
            aspectRatio: 'Aspect ratio',
            aspectRatioFree: 'Free',
            aspectRatioPresetOption: '{{name}}, {{ratio}}, {{size}}',
            height: 'Height',
            lockAspectRatio: 'Lock aspect ratio',
            megapixelsValue: '{{value}} MP',
            providerSize: 'Provider size',
            resetSize: 'Reset size',
            resetSizeModelDefaults: 'Model defaults',
            resetSizeOptimalForCurrentRatio: 'Optimal size for current ratio',
            resetSizeOptimalForRatio: 'Optimal size for {{ratio}}',
            resetSizeSquareNative: 'Square at native size',
            resetToBaseDefault: 'Reset to {{base}} default',
            resetToModelAndBaseDefault: 'Reset to model and {{base}} default',
            resetToModelDefault: 'Reset to model default',
            sectionMenu: {
              resetSize: 'Reset size to model defaults',
              resetSizeBase: 'Reset size to {{base}} defaults',
              resetSizeMixed: 'Reset size to model and {{base}} defaults',
            },
            size: 'Size',
            sizePreviewHandle: 'Resize',
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

const render = async (settings: Partial<GenerateSettings>, model: GenerateModelConfig | null = sd1Model) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const draft = createExternalStoreCore({ ...getDefaultGenerateSettings(model ?? undefined), ...settings });
  const onCommit = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));
  const onResetSize = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateDimensionFields
            draft={draft}
            projectId="project"
            selectedModel={model ?? undefined}
            onCommit={onCommit}
            onResetSize={onResetSize}
          />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return { onCommit, onResetSize };
};

const lockButton = () =>
  host?.querySelector<HTMLButtonElement>(
    'button[aria-label="Lock aspect ratio"], button[aria-label="Unlock aspect ratio"]'
  ) ?? null;
const bracket = () => host?.querySelector('[data-part="bracket"]')?.parentElement ?? null;
const ratioTrigger = () => page.getByRole('button', { name: /^Aspect ratio/ });
const chooseRatio = async (name: string) => {
  await userEvent.click(ratioTrigger());
  await userEvent.click(page.getByRole(name.includes(',') ? 'button' : 'radio', { exact: true, name }));
};

const unmountAndReset = async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
};

afterEach(async () => {
  harness.aspectRatioPresets = [];
  harness.invocationSourceId = 'generate';
  await unmountAndReset();
});

describe('GenerateDimensionFields size preview handle', () => {
  const pointer = (type: string, x: number, y: number) =>
    new PointerEvent(type, { bubbles: true, buttons: 1, clientX: x, clientY: y, pointerId: 1 });

  it('offers a 16px target and drops the dragged size when the browser cancels the drag', async () => {
    const { onCommit } = await render({ height: 512, width: 512 });
    const handle = host!.querySelector<HTMLButtonElement>('button[aria-label="Resize"]')!;
    const rect = handle.getBoundingClientRect();

    expect(rect.width).toBeGreaterThanOrEqual(16);
    expect(rect.height).toBeGreaterThanOrEqual(16);

    await settle(() => handle.dispatchEvent(pointer('pointerdown', rect.x, rect.y)));
    await settle(() => handle.dispatchEvent(pointer('pointermove', rect.x + 20, rect.y + 20)));
    expect(host?.textContent).not.toContain('512×512');

    await settle(() => handle.dispatchEvent(pointer('pointercancel', rect.x + 20, rect.y + 20)));

    expect(host?.textContent).toContain('512×512');
    expect(onCommit).not.toHaveBeenCalled();
  });
});

describe('GenerateDimensionFields provider size', () => {
  const geminiImage: GenerateModelConfig = {
    ...gptImage,
    provider_id: 'gemini',
    capabilities: {
      modes: ['txt2img'],
      resolution_presets: [
        { aspect_ratio: '1:1', height: 1024, image_size: '1K', label: '1:1 (1K)', width: 1024 },
        { aspect_ratio: '16:9', height: 1152, image_size: '2K', label: '16:9 (2K)', width: 2048 },
      ],
    },
  };
  const providerTrigger = () => {
    const providerLabel = [...(host?.querySelectorAll('label') ?? [])].find(
      (label) => label.textContent === 'Provider size'
    );

    return host?.querySelector<HTMLButtonElement>(`[role="combobox"][aria-labelledby~="${providerLabel?.id}"]`) ?? null;
  };

  it('commits the hosted preset through the dimension panel with its dimensions and name', async () => {
    const { onCommit } = await render({ externalImageSize: null, height: 1024, width: 1024 }, geminiImage);

    expect(providerTrigger()?.textContent).toContain('1:1 (1K)');
    await settle(() => providerTrigger()?.click());
    await settle(() => document.querySelector<HTMLElement>('[role="option"][data-value="16:9|2K"]')?.click());

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2048 / 1152,
      externalImageSize: '2K',
      height: 1152,
      width: 2048,
    });
  });
});

describe('GenerateDimensionFields aspect ratio', () => {
  it('reshapes the size to a chosen preset and locks it', async () => {
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 1024, width: 1024 });

    expect(lockButton()?.getAttribute('aria-label')).toBe('Lock aspect ratio');
    expect(bracket()?.hasAttribute('data-locked')).toBe(false);

    await chooseRatio('16:9');

    expect(onCommit).toHaveBeenCalledWith({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1368,
    });
  });

  it('swaps width and height and mirrors the preset', async () => {
    const { onCommit } = await render({
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

describe('GenerateDimensionFields Reset size', () => {
  const resetSizeButton = () => host?.querySelector<HTMLButtonElement>('button[aria-label="Reset size"]') ?? null;
  const menuItems = () => [...document.querySelectorAll<HTMLElement>('[role="menuitemradio"]')];
  const menuItem = (label: string) => menuItems().find((item) => item.textContent?.startsWith(label)) ?? null;
  const openMenu = () => settle(() => resetSizeButton()?.click());
  // Zag selects on a real pointer release, which a synthetic click() does not send.
  const choose = async (label: string) => {
    const item = menuItem(label);
    if (!item) {
      throw new Error(`no "${label}" choice`);
    }
    await userEvent.click(item);
    await settle(() => undefined);
  };

  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  afterEach(() => {
    resetArchitectureCapabilities();
  });

  it('lists each choice with the size it sets and resets to a locked square at the native size', async () => {
    const { onCommit, onResetSize } = await render({
      aspectRatioId: '2:3',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2 / 3,
      height: 768,
      width: 512,
    });

    await openMenu();

    // SD 1.5 is native at 512 on an 8px grid: 512² reshaped to 2:3 is 418×627, snapped to 416×624. The optimal size for
    // the ratio comes first, as the choice most reached for; this model sets no size of its own.
    expect(menuItems().map((item) => item.textContent)).toEqual([
      'Optimal size for 2:3416×624',
      'Square at native size512×512',
    ]);

    await choose('Square at native size');

    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: '1:1',
      aspectRatioIsLocked: true,
      aspectRatioValue: 1,
      height: 512,
      width: 512,
    });
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('lands the keyboard on the applied choice, and choosing it again changes nothing', async () => {
    const { onResetSize } = await render({
      aspectRatioId: '1:1',
      aspectRatioIsLocked: true,
      aspectRatioValue: 1,
      height: 512,
      width: 512,
    });

    resetSizeButton()?.focus();
    await userEvent.keyboard('{Enter}');
    await vi.waitFor(() => expect(document.activeElement?.getAttribute('role')).toBe('menu'));
    await userEvent.keyboard('{ArrowDown}');
    const square = menuItem('Square at native size');
    await vi.waitFor(() => expect(square?.hasAttribute('data-highlighted')).toBe(true));
    expect(document.activeElement?.getAttribute('aria-activedescendant')).toBe(square?.id);

    await userEvent.keyboard('{Enter}');

    await vi.waitFor(() => expect(menuItems()).toHaveLength(0));
    expect(onResetSize).not.toHaveBeenCalled();
  });
});

describe('GenerateDimensionFields title defaults', () => {
  const title = () => host!.querySelector<HTMLElement>('.generate-section > [data-part="trigger"]')!;
  const isMarked = () => title().querySelector('[data-section-changed]') !== null;
  const openMenu = async () => {
    await act(() => {
      title().dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.element(page.getByRole('menu')).toBeVisible();
  };
  const resetItem = () => page.getByRole('menuitem', { name: 'Reset size to model defaults' });

  it('marks a size off the model default and resets it from the title menu', async () => {
    const portraitModel: MainModelConfig = { ...sd1Model, default_settings: { height: 768, width: 640 } };
    const { onResetSize } = await render({ height: 512, width: 512 }, portraitModel);

    expect(isMarked()).toBe(true);
    await openMenu();
    await userEvent.click(resetItem());

    expect(onResetSize).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ height: 768, width: 640 }));
    await vi.waitFor(() => expect(isMarked()).toBe(false));
  });
});
