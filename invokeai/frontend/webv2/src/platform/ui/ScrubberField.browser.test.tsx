import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { getHintCard, openHintCard } from './hints/featureHint.testing';
import { type FeatureHintsAdapter, FeatureHintsProvider } from './hints/hintsContext';
import { ScrubberField, type ScrubberFieldProps } from './ScrubberField';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: { common: { scrubber: { editValue: 'Edit {{label}}: {{value}}' } } } } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const TRACK_INSET_PX = 6;
const HOST_WIDTH_PX = 400;

const mount = async (props: Partial<ScrubberFieldProps> = {}, width = HOST_WIDTH_PX) => {
  const onChange = vi.fn();

  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ScrubberField label="Steps" max={100} min={0} step={1} value={30} onChange={onChange} {...props} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const frame = host.querySelector<HTMLDivElement>('[data-scope="scrubber"]');
  const slider = host.querySelector<HTMLDivElement>('[role="slider"]');

  if (!frame || !slider) {
    throw new Error('ScrubberField did not render');
  }

  return { frame, onChange, slider };
};

/** Client X for a fraction of the track, which is inset from the frame edges. */
const trackX = (frame: HTMLElement, fraction: number): number => {
  const rect = frame.getBoundingClientRect();

  return rect.left + TRACK_INSET_PX + fraction * (rect.width - TRACK_INSET_PX * 2);
};

const pointer = (
  target: EventTarget,
  type: 'pointercancel' | 'pointerdown' | 'pointermove' | 'pointerup',
  init: PointerEventInit
) => act(() => target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, ...init })));

/** A caller that owns the value, so repeated keys step from what the previous step produced. */
const ControlledScrubber = (props: Omit<ScrubberFieldProps, 'value'> & { initial: number }) => {
  const { initial, onChange, ...rest } = props;
  const [value, setValue] = useState(initial);
  const handleChange = useCallback(
    (next: number) => {
      setValue(next);
      onChange(next);
    },
    [onChange]
  );

  return <ScrubberField {...rest} value={value} onChange={handleChange} />;
};

const mountControlled = async (initial: number) => {
  const events: string[] = [];
  const onChange = vi.fn((value: number) => events.push(`change:${value}`));
  const onChangeEnd = vi.fn((value: number) => events.push(`end:${value}`));

  host = document.createElement('div');
  host.style.width = `${HOST_WIDTH_PX}px`;
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ControlledScrubber
            defaultValue={50}
            initial={initial}
            label="Steps"
            max={100}
            min={0}
            step={1}
            onChange={onChange}
            onChangeEnd={onChangeEnd}
          />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const frame = host.querySelector<HTMLDivElement>('[data-scope="scrubber"]')!;
  const slider = host.querySelector<HTMLDivElement>('[role="slider"]')!;

  return { events, frame, onChangeEnd, slider };
};

const valueButton = () => host?.querySelector<HTMLButtonElement>('button[data-part="value"]') ?? null;
const editor = () => host?.querySelector<HTMLInputElement>('input[data-part="value"]') ?? null;

describe('ScrubberField', () => {
  it('exposes a labelled slider with the formatted value as its text', async () => {
    const { slider } = await mount({ formatValue: (value) => `${value}%` });
    const label = host?.querySelector(`#${CSS.escape(slider.getAttribute('aria-labelledby') ?? '')}`);

    expect(label?.textContent).toBe('Steps');
    expect(slider.getAttribute('aria-valuenow')).toBe('30');
    expect(slider.getAttribute('aria-valuetext')).toBe('30%');
    expect(valueButton()?.textContent).toBe('30%');
    // The name carries the visible value (WCAG 2.5.3).
    await expect.element(valueButton()!).toHaveAccessibleName('Edit Steps: 30%');
  });

  it('scrubs relative to the current value from wherever the press lands, snapped to the step', async () => {
    const { frame, onChange } = await mount({ step: 5 });

    // Like a native iOS slider: pressing away from the thumb does not move it there.
    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5) });

    expect(onChange).not.toHaveBeenCalled();
    expect(frame.hasAttribute('data-dragging')).toBe(true);

    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.73) });

    expect(onChange).toHaveBeenLastCalledWith(55);

    await pointer(window, 'pointermove', { clientX: trackX(frame, 2) });

    expect(onChange).toHaveBeenLastCalledWith(100);

    await pointer(window, 'pointerup', { clientX: trackX(frame, 2) });

    expect(frame.hasAttribute('data-dragging')).toBe(false);

    await pointer(window, 'pointermove', { clientX: trackX(frame, 0) });

    expect(onChange).toHaveBeenLastCalledWith(100);
  });

  it('scrubs relative to the current value at a tenth of the sensitivity while Shift is held', async () => {
    const { frame, onChange } = await mount();

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.9), shiftKey: true });

    expect(onChange).not.toHaveBeenCalled();

    // A full-track sweep in fine mode covers a tenth of the range.
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.4), shiftKey: true });

    expect(onChange).toHaveBeenLastCalledWith(25);
    await pointer(window, 'pointerup', { clientX: trackX(frame, 0.4) });
  });

  it('snaps to the nearest stop while Alt is held', async () => {
    const { frame, onChange } = await mount({ marks: [0, 20, 50, 100] });

    await pointer(frame, 'pointerdown', { altKey: true, clientX: trackX(frame, 0.3) });
    await pointer(window, 'pointermove', { altKey: true, clientX: trackX(frame, 0.45) });

    expect(onChange).toHaveBeenLastCalledWith(50);

    await pointer(window, 'pointermove', { altKey: true, clientX: trackX(frame, 0.8) });

    expect(onChange).toHaveBeenLastCalledWith(100);
    await pointer(window, 'pointerup', { clientX: trackX(frame, 0.8) });
  });

  it('steps from the keyboard, ×10 with Shift or Page keys, and clamps typed-range bounds', async () => {
    const { onChange, slider } = await mount({ inputMax: 1000, step: 0.5, value: 99.5 });

    await act(async () => {
      slider.focus();
      await userEvent.keyboard('{ArrowRight}');
    });

    expect(onChange).toHaveBeenLastCalledWith(100);

    await act(() => userEvent.keyboard('{Shift>}{ArrowUp}{/Shift}'));

    expect(onChange).toHaveBeenLastCalledWith(104.5);

    await act(() => userEvent.keyboard('{PageDown}'));

    expect(onChange).toHaveBeenLastCalledWith(94.5);

    await act(() => userEvent.keyboard('{Home}'));

    expect(onChange).toHaveBeenLastCalledWith(0);

    await act(() => userEvent.keyboard('{End}'));

    expect(onChange).toHaveBeenLastCalledWith(100);
  });

  it('opens the editor from the value button, commits on Enter, and returns focus to the slider', async () => {
    const { onChange, slider } = await mount({ inputMax: 500 });

    await act(() => {
      valueButton()?.click();
    });

    const input = editor();

    expect(document.activeElement).toBe(input);
    expect(input?.value).toBe('30');
    expect(valueButton()).toBeNull();
    expect(host?.querySelector('[data-part="label"]')?.checkVisibility()).toBe(true);

    await act(() => userEvent.keyboard('250{Enter}'));

    expect(onChange).toHaveBeenLastCalledWith(250);
    expect(editor()).toBeNull();
    expect(document.activeElement).toBe(slider);
  });

  it('leaves focus where a blur sent it and only returns to the slider on Enter or Escape', async () => {
    const { onChange, slider } = await mount({ inputMax: 500 });
    const next = document.createElement('button');

    host?.append(next);
    await act(() => {
      valueButton()?.click();
    });
    await act(() => userEvent.keyboard('40'));
    await act(() => {
      next.focus();
    });

    expect(onChange).toHaveBeenLastCalledWith(40);
    expect(editor()).toBeNull();
    expect(document.activeElement).toBe(next);

    await act(() => {
      slider.focus();
    });
    await act(() => userEvent.keyboard('{Enter}{Escape}'));

    expect(document.activeElement).toBe(slider);

    // A blur with no destination (window deactivated, click on nothing focusable) keeps the tab stop.
    await act(() => userEvent.keyboard('{Enter}'));
    await act(() => {
      editor()?.blur();
    });

    expect(editor()).toBeNull();
    expect(document.activeElement).toBe(slider);
  });

  it('consumes the keys it handles before they reach window-level shortcuts', async () => {
    // No default: Backspace/Delete restore nothing but are still the slider's keys.
    const { onChange, slider } = await mount();
    const seen: string[] = [];
    const listener = (event: KeyboardEvent) => {
      seen.push(event.key);
    };

    window.addEventListener('keydown', listener);

    try {
      await act(async () => {
        slider.focus();
        await userEvent.keyboard('{ArrowRight}{Home}{Delete}2');
      });

      expect(editor()?.value).toBe('2');
      expect(seen).toEqual([]);
      expect(onChange).toHaveBeenLastCalledWith(0);

      await act(() => userEvent.keyboard('{Escape}'));
      await act(() => userEvent.keyboard('q'));

      expect(seen).toEqual(['Escape', 'q']);
    } finally {
      window.removeEventListener('keydown', listener);
    }
  });

  it('restores the default on double-click or Backspace, and only when one is given', async () => {
    const { frame, onChange, slider } = await mount({ defaultValue: 20 });

    await act(() => frame.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, clientX: trackX(frame, 0.5) })));

    expect(onChange).toHaveBeenLastCalledWith(20);

    await act(async () => {
      slider.focus();
      await userEvent.keyboard('{Backspace}');
    });

    expect(onChange).toHaveBeenCalledTimes(2);
    expect(onChange).toHaveBeenLastCalledWith(20);

    await act(() => root?.unmount());
    host?.remove();

    const noDefault = await mount();

    await act(() =>
      noDefault.frame.dispatchEvent(
        new MouseEvent('dblclick', { bubbles: true, clientX: trackX(noDefault.frame, 0.5) })
      )
    );

    expect(noDefault.onChange).not.toHaveBeenCalled();
  });

  it('lets a touch pan the panel until it moves sideways', async () => {
    const { frame, onChange } = await mount();

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5), pointerType: 'touch' });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.5) + 4, pointerType: 'touch' });

    expect(onChange).not.toHaveBeenCalled();

    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.7), pointerType: 'touch' });

    expect(onChange).toHaveBeenLastCalledWith(50);
    await pointer(window, 'pointerup', { pointerType: 'touch' });
  });

  it('keeps one tab stop per field', async () => {
    await mount();

    expect(valueButton()?.getAttribute('tabindex')).toBe('-1');
  });

  it('describes the slider with the error line and flags the frame invalid', async () => {
    const { frame, slider } = await mount({ error: 'Too high', helpText: 'Ignored while erroring' });
    const described = host?.querySelector(`#${CSS.escape(slider.getAttribute('aria-describedby') ?? '')}`);

    expect(described?.textContent).toBe('Too high');
    expect(described?.getAttribute('role')).toBe('alert');
    expect(frame.hasAttribute('data-invalid')).toBe(true);
  });

  describe('gesture end', () => {
    it('ends a drag once with its final value on release, and not for a press that never moved', async () => {
      const { events, frame } = await mountControlled(30);

      await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5) });
      await pointer(window, 'pointerup', { clientX: trackX(frame, 0.5) });

      expect(events).toEqual([]);

      await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5) });
      await pointer(window, 'pointermove', { clientX: trackX(frame, 0.6) });
      await pointer(window, 'pointermove', { clientX: trackX(frame, 0.7) });

      expect(events).toEqual(['change:40', 'change:50']);

      await pointer(window, 'pointercancel', { clientX: trackX(frame, 0.7) });

      expect(events).toEqual(['change:40', 'change:50', 'end:50']);
    });

    it('ends a held step key once on release, and a tapped key per tap', async () => {
      const { events, slider } = await mountControlled(30);

      await act(async () => {
        slider.focus();
        await userEvent.keyboard('{ArrowRight>3/}');
      });

      expect(events).toEqual(['change:31', 'change:32', 'change:33', 'end:33']);

      events.length = 0;
      await act(() => userEvent.keyboard('{ArrowLeft}{ArrowLeft}'));

      expect(events).toEqual(['change:32', 'end:32', 'change:31', 'end:31']);
    });
  });
});

describe('ScrubberField log scale and value-dependent steps', () => {
  /** Where an element's centre sits along the track, 0–1. */

  const LOG = { max: 100, min: 1, scale: 'log', step: 1, value: 10 } as const;

  it('scrubs along the track by ratio, finely with Shift, snapped to the step in real units', async () => {
    const { frame, onChange } = await mount(LOG);

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5) });
    // A quarter-track further multiplies by 10^0.5.
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.75) });

    expect(onChange).toHaveBeenLastCalledWith(32);
    await pointer(window, 'pointerup', { clientX: trackX(frame, 0.75) });

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5), shiftKey: true });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 1), shiftKey: true });

    // Half a track at a tenth of the sensitivity: 10 × 10^0.1.
    expect(onChange).toHaveBeenLastCalledWith(13);
    await pointer(window, 'pointerup', { clientX: trackX(frame, 1) });
  });
});

describe('ScrubberField hints', () => {
  const HINT_RESOURCES = {
    hints: { steps: { heading: 'Steps', paragraphs: ['Number of steps that will be performed.'] } },
  };

  i18n.addResourceBundle('en', 'translation', HINT_RESOURCES, true, false);

  const HINTS_ON: FeatureHintsAdapter = { enabled: true, onDisable: null };
  const HINTS_OFF: FeatureHintsAdapter = { enabled: false, onDisable: null };
  /** Inset from the parked pointer, and narrow enough to leave room for the card beside the field. */
  const HOST_CSS = 'width:120px;margin:120px 0 0 8px';

  const mountWithHints = async (enabled: boolean) => {
    const onChange = vi.fn();

    host = document.createElement('div');
    host.style.cssText = HOST_CSS;
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <FeatureHintsProvider adapter={enabled ? HINTS_ON : HINTS_OFF}>
              <ScrubberField hint="steps" label="Steps" max={100} min={0} step={1} value={30} onChange={onChange} />
            </FeatureHintsProvider>
          </ChakraProvider>
        </I18nextProvider>
      );
    });

    const frame = host.querySelector<HTMLDivElement>('[data-scope="scrubber"]')!;
    const slider = host.querySelector<HTMLDivElement>('[role="slider"]')!;
    const label = host.querySelector<HTMLElement>('[data-part="label"]')!;

    return { frame, label, onChange, slider };
  };

  it('opens the hint card from the label beside the field, keeping the slider named by the label', async () => {
    const { frame, label, slider } = await mountWithHints(true);
    const card = await openHintCard(label.querySelector('[data-part="trigger"]')!);

    expect(getHintCard()?.textContent).toContain('Number of steps that will be performed.');
    // Beside the frame, not over the track and value the hint explains.
    expect(card.left).toBeGreaterThanOrEqual(frame.getBoundingClientRect().right);
    expect(document.getElementById(slider.getAttribute('aria-labelledby') ?? '')?.textContent).toBe('Steps');
  });
});
