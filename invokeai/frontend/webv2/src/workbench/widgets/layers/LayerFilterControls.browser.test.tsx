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

const render = async () => {
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
            filterType="canny_edge_detection"
            focusFilter={false}
            parts="params"
            settings={SETTINGS}
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
