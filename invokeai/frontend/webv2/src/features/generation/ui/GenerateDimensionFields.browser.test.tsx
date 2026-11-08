/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateModelConfig, GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
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
    // Stands in for the canvas's render-size slot, which lays out the Size section around its controls.
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
            resetToModelDefault: 'Reset to model default',
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

const render = async (settings: Partial<GenerateSettings>, model: GenerateModelConfig = sd1Model) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const draft = createExternalStoreCore({ ...getDefaultGenerateSettings(model), ...settings });
  // Commits and resets land in the draft, as the panel's draft would show them.
  const onCommit = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));
  const onResetSize = vi.fn((patch: Partial<GenerateSettings>) => draft.patchSnapshot(patch));

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateDimensionFields
            draft={draft}
            projectId="project"
            selectedModel={model}
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
  // Built-ins are radios in the ratio grid; saved ratios are rows named "name, ratio, size".
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

describe('GenerateDimensionFields canvas render size', () => {
  it('lays out the render-size controls and the resized badge for a local model on the canvas', async () => {
    harness.invocationSourceId = 'canvas';
    await render({ height: 512, width: 512 });

    expect(host?.querySelector('[data-testid="render-size-controls"]')).not.toBeNull();
    expect(host?.textContent).toContain('512×512 → 1024×1024');
  });

  it('shows a hosted provider only the frame, which is the size the provider is asked for', async () => {
    harness.invocationSourceId = 'canvas';
    await render({ height: 512, width: 512 }, gptImage);

    expect(host?.querySelector('[data-testid="render-size-controls"]')).toBeNull();
    expect(host?.textContent).not.toContain('→');
    expect(host?.textContent).toContain('512×512');
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

  it('selects the preset from the panel dimensions, breaking a same-size tie by the stored name', async () => {
    await render({ externalImageSize: null, height: 1152, width: 2048 }, geminiImage);
    expect(providerTrigger()?.textContent).toContain('16:9 (2K)');

    await settle(() => root?.unmount());
    host?.remove();

    const duplicateSizes: GenerateModelConfig = {
      ...geminiImage,
      capabilities: {
        modes: ['txt2img'],
        resolution_presets: [
          ...(geminiImage.capabilities?.resolution_presets ?? []),
          { aspect_ratio: '1:1', height: 1024, image_size: 'HD', label: '1:1 (HD)', width: 1024 },
        ],
      },
    };
    await render({ externalImageSize: 'HD', height: 1024, width: 1024 }, duplicateSizes);
    expect(providerTrigger()?.textContent).toContain('1:1 (HD)');
  });

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

  it('shows the lock engaged while a preset is selected and unlocking returns to Free', async () => {
    // A preset implies ratio lock even when the stored flag is false.
    const { onCommit } = await render({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: false,
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1368,
    });

    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio 16:9');
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
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 800, width: 1024 });

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
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 768, width: 1024 });

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
  it('keeps the pixel count when choosing a saved ratio and stores it as a locked Free ratio', async () => {
    harness.aspectRatioPresets = [{ height: 1, id: 'cinema', label: 'Cinema', width: 2.39 }];
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 1024, width: 1024 });

    // 1024² at 2.39:1 is 1583.1×662.4, snapped to 1584×664: the same pixel count, never the model's optimum.
    await chooseRatio('Cinema, 2.39:1, 1584×664');

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2.39,
      height: 664,
      width: 1584,
    });
    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio Cinema 2.39:1');
  });

  it('stores a saved ratio equal to a built-in under the built-in id, still named by the saved ratio', async () => {
    harness.aspectRatioPresets = [{ height: 9, id: 'wide', label: 'Wide', width: 16 }];
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 1024, width: 1024 });

    await chooseRatio('Wide, 16:9, 1368×768');

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: '16:9',
      aspectRatioIsLocked: true,
      aspectRatioValue: 16 / 9,
      height: 768,
      width: 1368,
    });
    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio Wide 16:9');
  });

  it('keeps a locked extreme ratio at the size limit, scaling both sides', async () => {
    harness.aspectRatioPresets = [{ height: 1, id: 'banner', label: 'Banner', width: 7 }];
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 2048, width: 2048 });

    // 2048² at 7:1 is 5418.6×774.1; scaled to fit 4096 wide it is 4096×585.1, snapped to 4096×584 (7.01:1).
    await chooseRatio('Banner, 7:1, 4096×584');

    expect(onCommit).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 7,
      height: 584,
      width: 4096,
    });
  });

  it('swaps a saved ratio into the saved ratio that is its inverse', async () => {
    harness.aspectRatioPresets = [
      { height: 5, id: 'banner', label: 'Banner', width: 12 },
      { height: 12, id: 'skyscraper', label: 'Skyscraper', width: 5 },
    ];
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: true, aspectRatioValue: 2.4, height: 640, width: 1536 });
    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio Banner 12:5');

    await settle(() => host?.querySelector<HTMLButtonElement>('button[aria-label="Swap width and height"]')?.click());

    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio Skyscraper 5:12');
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

    // SD 1.5 is native at 512 on an 8px grid: 512² reshaped to 2:3 is 418×627, snapped to 416×624.
    // The optimal size for the ratio comes first, as the choice most reached for; this model sets no size of its own.
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

  it('keeps an unlocked free ratio when resetting to its optimal size', async () => {
    const { onResetSize } = await render({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      height: 400,
      width: 800,
    });

    await openMenu();
    // 512² at 2:1 is 724×362, snapped to 728×360.
    expect(menuItem('Optimal size for current ratio')?.textContent).toBe('Optimal size for current ratio728×360');

    await choose('Optimal size for current ratio');

    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({ height: 360, width: 728 });
  });

  it('names a locked free ratio by small whole terms, else by decimals in the app locale', async () => {
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: true, aspectRatioValue: 2.5, height: 400, width: 1000 });
    await openMenu();
    // 512² at 5:2 is 809.5×323.8, snapped to 808×320.
    expect(menuItem('Optimal size for 5:2')?.textContent).toBe('Optimal size for 5:2808×320');
    await unmountAndReset();

    i18n.addResourceBundle('de', 'translation', i18n.getResourceBundle('en', 'translation'));
    await i18n.changeLanguage('de');
    try {
      await render({
        aspectRatioId: 'Free',
        aspectRatioIsLocked: true,
        aspectRatioValue: 2.39,
        height: 400,
        width: 960,
      });
      await openMenu();

      // 512² at 2.39:1 is 791.5×331.2, snapped to 792×328.
      expect(menuItem('Optimal size for 2,39:1')?.textContent).toBe('Optimal size for 2,39:1792×328');
    } finally {
      await i18n.changeLanguage('en');
    }
  });

  it('names a locked ratio by the saved ratio it matches', async () => {
    harness.aspectRatioPresets = [{ height: 5, id: 'banner', label: 'Banner', width: 12 }];
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: true, aspectRatioValue: 2.4, height: 640, width: 1536 });
    await openMenu();

    // 512² at 12:5 is 793.3×330.5, snapped to 792×328.
    expect(menuItem('Optimal size for Banner')?.textContent).toBe('Optimal size for Banner792×328');
  });

  it('checks the applied choice, keeps it reachable, and drops an optimal size that is the native square', async () => {
    await render({ aspectRatioId: '1:1', aspectRatioIsLocked: true, aspectRatioValue: 1, height: 512, width: 512 });

    await openMenu();

    const square = menuItem('Square at native size');
    expect(square?.getAttribute('aria-checked')).toBe('true');
    expect(square?.hasAttribute('aria-disabled')).toBe(false);
    expect(menuItems()).toHaveLength(1);
    await unmountAndReset();

    // The same size under a free ratio is not yet the locked square.
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: 1, height: 512, width: 512 });
    await openMenu();
    expect(menuItem('Square at native size')?.getAttribute('aria-checked')).toBe('false');
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

  it('offers model defaults only for a model that sets its own size', async () => {
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 512, width: 512 });
    await openMenu();
    expect(menuItem('Model defaults')).toBeNull();
    await unmountAndReset();

    const sizedModel: MainModelConfig = { ...sd1Model, default_settings: { height: 768, width: 640 } };
    const { onResetSize } = await render(
      { aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 512, width: 512 },
      sizedModel
    );
    await openMenu();
    expect(menuItem('Model defaults')?.textContent).toBe('Model defaults640×768');

    await choose('Model defaults');

    // An unlocked ratio stays unlocked, at the default size's ratio.
    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 640 / 768,
      height: 768,
      width: 640,
    });
  });

  const squareModel: MainModelConfig = { ...sd1Model, default_settings: { height: 1024, width: 1024 } };
  const widescreen = {
    aspectRatioId: '16:9',
    aspectRatioIsLocked: true,
    aspectRatioValue: 16 / 9,
    height: 768,
    width: 1368,
  } as const;

  it('locks a square model default at 1:1 when a non-square ratio was locked', async () => {
    const { onResetSize } = await render(widescreen, squareModel);
    await openMenu();

    await choose('Model defaults');

    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: '1:1',
      aspectRatioIsLocked: true,
      aspectRatioValue: 1,
      height: 1024,
      width: 1024,
    });
    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio 1:1');
    expect(lockButton()?.getAttribute('aria-label')).toBe('Unlock aspect ratio');
    expect(host?.textContent).toContain('1024×1024');
  });

  it('leaves the ratio unlocked when resetting an unlocked size to a square model default', async () => {
    const { onResetSize } = await render(
      { aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: 16 / 9, height: 768, width: 1368 },
      squareModel
    );
    await openMenu();

    await choose('Model defaults');

    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: false,
      aspectRatioValue: 1,
      height: 1024,
      width: 1024,
    });
    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio Free');
    expect(lockButton()?.getAttribute('aria-label')).toBe('Lock aspect ratio');
  });

  it('locks the default size ratio from the size field Reset to model default', async () => {
    const portraitModel: MainModelConfig = { ...sd1Model, default_settings: { height: 768, width: 640 } };
    const { onResetSize } = await render(widescreen, portraitModel);

    await userEvent.click(page.getByRole('slider', { name: 'Width' }), { button: 'right' });
    await userEvent.click(page.getByRole('menuitem', { name: 'Reset to model default' }));

    // 640×768 is 5:6, no built-in: a locked Free ratio.
    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({
      aspectRatioId: 'Free',
      aspectRatioIsLocked: true,
      aspectRatioValue: 640 / 768,
      height: 768,
      width: 640,
    });
    await expect.element(ratioTrigger()).toHaveAccessibleName('Aspect ratio 5:6');
  });

  it('orders the optimal size first, then the model defaults, then the native square', async () => {
    const sizedModel: MainModelConfig = { ...sd1Model, default_settings: { height: 768, width: 640 } };
    await render(
      { aspectRatioId: '2:3', aspectRatioIsLocked: true, aspectRatioValue: 2 / 3, height: 768, width: 512 },
      sizedModel
    );
    await openMenu();

    expect(menuItems().map((item) => item.textContent)).toEqual([
      'Optimal size for 2:3416×624',
      'Model defaults640×768',
      'Square at native size512×512',
    ]);
  });

  it('opens from the keyboard and applies the highlighted choice', async () => {
    const { onResetSize } = await render({
      aspectRatioId: '2:3',
      aspectRatioIsLocked: true,
      aspectRatioValue: 2 / 3,
      height: 768,
      width: 512,
    });

    resetSizeButton()?.focus();
    await userEvent.keyboard('{Enter}');
    await vi.waitFor(() => expect(document.activeElement?.getAttribute('role')).toBe('menu'));
    // Arrows down to the native square by its name, wherever the menu lists it.
    const square = menuItem('Square at native size');
    for (let step = 0; step < menuItems().length && !square?.hasAttribute('data-highlighted'); step += 1) {
      await userEvent.keyboard('{ArrowDown}');
    }
    expect(square?.hasAttribute('data-highlighted')).toBe(true);
    await userEvent.keyboard('{Enter}');

    await vi.waitFor(() =>
      expect(onResetSize).toHaveBeenCalledExactlyOnceWith({
        aspectRatioId: '1:1',
        aspectRatioIsLocked: true,
        aspectRatioValue: 1,
        height: 512,
        width: 512,
      })
    );
  });

  it('has no footer optimal-size button', async () => {
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 384, width: 384 });

    expect(host?.textContent).not.toContain('Set optimal size');
  });
});
