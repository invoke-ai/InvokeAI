/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, type ComponentProps } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoLengthControls } from './VideoLengthControls';

/**
 * Auto duration turns Frames from a length into a ceiling. These pin what the panel says about
 * that, because the number that sizes the run's memory must stay visible and editable.
 */

const i18n = createInstance();

void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          video: {
            autoDuration: 'Auto duration',
            autoDurationCeiling: 'Up to {{frames}} frames ({{seconds}} s).',
            autoDurationHelp: 'Head chooses the length.',
            autoDurationTooShort: 'Too short to choose.',
            autoDurationUnavailable: 'This mode sets its own length.',
            frames: 'Frames',
            maxFrames: 'Max frames',
          },
        },
      },
    },
  },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const BASE: ComponentProps<typeof VideoLengthControls> = {
  autoDuration: false,
  autoDurationActive: false,
  autoDurationCeiling: null,
  autoDurationSupported: true,
  durationText: '5.0 s',
  framesSlider: { inputMax: 481, max: 241, min: 9, step: 8 },
  hasDurationHead: true,
  numFrames: 121,
  numFramesFromClip: false,
  onAutoDurationChange: vi.fn(),
  onNumFramesChange: vi.fn(),
};

describe('VideoLengthControls', () => {
  let host: HTMLDivElement;
  let root: Root;
  const mount = (props: Partial<ComponentProps<typeof VideoLengthControls>>) =>
    act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <VideoLengthControls {...BASE} {...props} />
          </ChakraProvider>
        </I18nextProvider>
      );
    });
  const autoSwitch = () => host.querySelector<HTMLInputElement>('input[type="checkbox"]');
  const frames = () => host.querySelector<HTMLElement>('[role="slider"]');
  const framesLabel = () =>
    host.querySelector(`#${CSS.escape(frames()?.getAttribute('aria-labelledby') ?? '')}`)?.textContent;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('offers no switch without a duration head', async () => {
    await mount({ hasDurationHead: false });

    expect(autoSwitch()).toBeNull();
    expect(host.textContent).toContain('Frames');
  });

  it('labels the switch and turns Frames into an editable ceiling that states the range', async () => {
    await mount({
      autoDuration: true,
      autoDurationActive: true,
      autoDurationCeiling: { frames: 121, seconds: 121 / 24 },
    });

    const input = autoSwitch();
    expect(input?.checked).toBe(true);
    expect(input?.labels?.[0]?.textContent).toBe('Auto duration');
    expect(framesLabel()).toBe('Max frames');
    expect(host.textContent).toContain('Up to 121 frames (5.0 s).');
    // The ceiling sizes the run's memory, so it stays editable.
    expect(frames()?.getAttribute('aria-disabled')).toBeNull();
  });

  it('says when the ceiling is too short for the head to choose', async () => {
    await mount({ autoDuration: true, autoDurationActive: true, autoDurationCeiling: null, numFrames: 17 });

    expect(host.textContent).toContain('Too short to choose.');
  });

  it('disables the switch, with the reason, in a mode that sets its own length', async () => {
    await mount({ autoDuration: true, autoDurationSupported: false, numFramesFromClip: true });

    expect(autoSwitch()?.disabled).toBe(true);
    expect(autoSwitch()?.checked).toBe(false);
    expect(host.textContent).toContain('This mode sets its own length.');
    expect(framesLabel()).toBe('Frames');
    expect(frames()?.getAttribute('aria-disabled')).toBe('true');
  });

  it('reports a toggle to its owner', async () => {
    const onAutoDurationChange = vi.fn();
    await mount({ onAutoDurationChange });

    await act(() => autoSwitch()?.click());

    expect(onAutoDurationChange).toHaveBeenCalledWith(expect.objectContaining({ checked: true }));
  });
});
