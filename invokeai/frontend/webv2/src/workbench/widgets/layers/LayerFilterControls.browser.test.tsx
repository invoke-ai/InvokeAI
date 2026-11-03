import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { LayerFilterControls } from './LayerFilterControls';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SETTINGS = { high_threshold: 200, low_threshold: 100 };
const ignoreFilterTypeChange = (): void => undefined;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const settle = (run: () => void = () => undefined) =>
  act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });

const render = async ({
  enumPresentation,
  filterType = 'canny_edge_detection',
  settings = SETTINGS,
}: {
  enumPresentation?: 'select' | 'segmented';
  filterType?: string;
  settings?: Record<string, unknown>;
} = {}) => {
  const onSettingsChange = vi.fn();
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <LayerFilterControls
            disabled={false}
            enumPresentation={enumPresentation}
            filterType={filterType}
            focusFilter={false}
            parts="params"
            settings={settings}
            onFilterTypeChange={ignoreFilterTypeChange}
            onSettingsChange={onSettingsChange}
          />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return onSettingsChange;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

// No locale resources: the field is labelled by its key's fallback, the parameter name.
const lowThreshold = () => page.getByRole('slider', { name: 'low_threshold' });

/** One drag on the low-threshold scrubber (a 0–255 track) through each offset, released at the last. */
const drag = async (offsets: number[], beforeRelease: () => void = () => undefined) => {
  const frame = lowThreshold().element().closest<HTMLElement>('[data-scope="scrubber"]')!;
  const rect = frame.getBoundingClientRect();
  const startX = rect.left + rect.width / 2;
  const xAt = (offset: number) => startX + offset * (rect.width - 20);
  const pointer = (target: EventTarget, type: string, x: number) =>
    settle(() => target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, clientX: x, pointerId: 1 })));
  await pointer(frame, 'pointerdown', startX);
  for (const offset of offsets) {
    await pointer(window, 'pointermove', xAt(offset));
  }
  beforeRelease();
  await pointer(window, 'pointerup', xAt(offsets.at(-1) ?? 0));
};

describe('LayerFilterControls number parameters', () => {
  it('shows a drag locally and changes the settings once, when it ends', async () => {
    const onSettingsChange = await render();

    await drag([0.1, 0.2], () => {
      expect(lowThreshold().element().getAttribute('aria-valuenow')).toBe('151');
      expect(onSettingsChange).not.toHaveBeenCalled();
    });

    expect(onSettingsChange).toHaveBeenCalledTimes(1);
    expect(onSettingsChange).toHaveBeenCalledWith({ ...SETTINGS, low_threshold: 151 });
  });

  it('leaves the settings alone when a drag returns to where it started', async () => {
    const onSettingsChange = await render();

    await drag([0.2, 0]);

    expect(onSettingsChange).not.toHaveBeenCalled();
    await expect.element(lowThreshold()).toHaveAttribute('aria-valuenow', '100');
  });

  it('rounds a typed value for an integer parameter', async () => {
    const onSettingsChange = await render();

    await act(async () => {
      (lowThreshold().element() as HTMLElement).focus();
      await userEvent.keyboard('{Enter}120.6{Enter}');
    });

    expect(onSettingsChange).toHaveBeenCalledTimes(1);
    expect(onSettingsChange).toHaveBeenCalledWith({ ...SETTINGS, low_threshold: 121 });
  });
});

describe('LayerFilterControls enum parameters', () => {
  const DEPTH_SETTINGS = { model_size: 'small_v2' };
  const modelSize = () => page.getByRole('radiogroup', { name: 'model_size' });

  it('shows a short enum as a labelled segmented row and reports clicks and arrow keys', async () => {
    const onSettingsChange = await render({
      enumPresentation: 'segmented',
      filterType: 'depth_anything_depth_estimation',
      settings: DEPTH_SETTINGS,
    });

    await expect.element(modelSize()).toBeVisible();
    expect(modelSize().getByRole('radio').all()).toHaveLength(4);
    await expect.element(modelSize().getByRole('radio', { name: 'small_v2' })).toBeChecked();

    await act(() => userEvent.click(modelSize().getByText('base', { exact: true })));
    expect(onSettingsChange).toHaveBeenLastCalledWith({ model_size: 'base' });

    // Arrow keys move from the checked segment, which a rerender would keep at small_v2.
    onSettingsChange.mockClear();
    await act(async () => {
      (modelSize().getByRole('radio', { name: 'small_v2' }).element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowLeft}');
    });
    expect(onSettingsChange).toHaveBeenLastCalledWith({ model_size: 'small' });
  });

  it('keeps a select for the default presentation and for enums too long to segment', async () => {
    await render({ filterType: 'depth_anything_depth_estimation', settings: DEPTH_SETTINGS });
    expect(document.querySelector('[role="radiogroup"]')).toBeNull();
    expect(document.querySelector('[data-scope="select"][data-part="trigger"]')).not.toBeNull();
    await act(() => root?.unmount());
    host?.remove();

    await render({ enumPresentation: 'segmented', filterType: 'adjust_image', settings: {} });
    expect(document.querySelector('[role="radiogroup"]')).toBeNull();
    expect(document.querySelector('[data-scope="select"][data-part="trigger"]')).not.toBeNull();
  });
});
