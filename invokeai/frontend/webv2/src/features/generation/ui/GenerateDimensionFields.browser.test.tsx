/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerateSettings, MainModelConfig } from '@features/generation/core/types';

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
import { userEvent } from 'vitest/browser';

import { GenerateDimensionFields } from './GenerateDimensionFields';

vi.mock('./GenerationUiContext', () => ({
  useGenerationQueueInsights: (select: (insights: unknown) => unknown) =>
    select({ secondsPerRun: null, seedHistory: [] }),
  useGenerationUi: () => ({
    project: { invocationSourceId: 'generate' },
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
            resetSize: 'Reset size',
            resetSizeModelDefaults: 'Model defaults',
            resetSizeOptimalForCurrentRatio: 'Optimal size for current ratio',
            resetSizeOptimalForRatio: 'Optimal size for {{ratio}}',
            resetSizeSquareNative: 'Square at native size',
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

const render = async (settings: Partial<GenerateSettings>, model: MainModelConfig = sd1Model) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onCommit = vi.fn();
  const onResetSize = vi.fn();

  await settle(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateDimensionFields
            draft={createExternalStoreCore({ ...getDefaultGenerateSettings(model), ...settings })}
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
const presetTrigger = () => host?.querySelector<HTMLButtonElement>('[role="combobox"]') ?? null;

const unmountAndReset = async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
};

afterEach(unmountAndReset);

describe('GenerateDimensionFields aspect ratio', () => {
  it('reshapes the size to a chosen preset and locks it', async () => {
    const { onCommit } = await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 1024, width: 1024 });

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
    const { onCommit } = await render({
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
});

describe('GenerateDimensionFields Reset size', () => {
  const resetSizeButton = () => host?.querySelector<HTMLButtonElement>('button[aria-label="Reset size"]') ?? null;
  const menuItems = () => [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')];
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
    expect(menuItems().map((item) => item.textContent)).toEqual([
      'Square at native size512×512',
      'Optimal size for 2:3416×624',
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

  it('names a locked free ratio in the app locale', async () => {
    i18n.addResourceBundle('de', 'translation', i18n.getResourceBundle('en', 'translation'));
    await i18n.changeLanguage('de');
    try {
      await render({
        aspectRatioId: 'Free',
        aspectRatioIsLocked: true,
        aspectRatioValue: 2.5,
        height: 400,
        width: 1000,
      });
      await openMenu();

      // 512² at 5:2 is 809.5×323.8, snapped to 808×320.
      expect(menuItem('Optimal size for 2,5:1')?.textContent).toBe('Optimal size for 2,5:1808×320');
    } finally {
      await i18n.changeLanguage('en');
    }
  });

  it('marks the applied choices current and disabled, and drops an optimal size that is the native square', async () => {
    await render({ aspectRatioId: '1:1', aspectRatioIsLocked: true, aspectRatioValue: 1, height: 512, width: 512 });

    await openMenu();

    const square = menuItem('Square at native size');
    expect(square?.getAttribute('aria-current')).toBe('true');
    expect(square?.getAttribute('aria-disabled')).toBe('true');
    expect(menuItems()).toHaveLength(1);
    await unmountAndReset();

    // The same size under a free ratio is not yet the locked square.
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: 1, height: 512, width: 512 });
    await openMenu();
    expect(menuItem('Square at native size')?.hasAttribute('aria-current')).toBe(false);
    expect(menuItem('Square at native size')?.hasAttribute('aria-disabled')).toBe(false);
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

    expect(onResetSize).toHaveBeenCalledExactlyOnceWith({ height: 768, width: 640 });
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
    await userEvent.keyboard('{ArrowDown}');
    await userEvent.keyboard('{ArrowDown}');
    await userEvent.keyboard('{Enter}');

    await vi.waitFor(() => expect(onResetSize).toHaveBeenCalledExactlyOnceWith({ height: 624, width: 416 }));
  });

  it('has no footer optimal-size button', async () => {
    await render({ aspectRatioId: 'Free', aspectRatioIsLocked: false, height: 384, width: 384 });

    expect(host?.textContent).not.toContain('Set optimal size');
  });
});
