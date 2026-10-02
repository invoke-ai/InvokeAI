import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ScrubberField, type ScrubberFieldProps } from './ScrubberField';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: { common: { scrubber: { editValue: 'Edit {{label}}' } } } } },
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

const TRACK_INSET_PX = 10;
const HOST_WIDTH_PX = 400;

const mount = async (props: Partial<ScrubberFieldProps> = {}) => {
  const onChange = vi.fn();

  host = document.createElement('div');
  host.style.width = `${HOST_WIDTH_PX}px`;
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
    expect(valueButton()?.getAttribute('aria-label')).toBe('Edit Steps');
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

  it('reaches either bound from a press on the far side of the thumb', async () => {
    const { frame, onChange } = await mount();

    // Right of the thumb: the value is not capped by where the press landed.
    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.9) });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 1.7) });

    expect(onChange).toHaveBeenLastCalledWith(100);
    await pointer(window, 'pointerup', { clientX: trackX(frame, 1.7) });

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.1) });
    await pointer(window, 'pointermove', { clientX: trackX(frame, -0.5) });

    expect(onChange).toHaveBeenLastCalledWith(0);
    await pointer(window, 'pointerup', { clientX: trackX(frame, -0.5) });
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

  it('ignores stops beyond the track while Alt is held', async () => {
    // Exclude out-of-range defaults from stops so Alt-drag cannot jump off-track.
    const { frame, onChange } = await mount({ defaultValue: 30, marks: [50], max: 10, value: 5 });

    await pointer(frame, 'pointerdown', { altKey: true, clientX: trackX(frame, 0.5) });
    await pointer(window, 'pointermove', { altKey: true, clientX: trackX(frame, 0.9) });

    expect(onChange).toHaveBeenLastCalledWith(9);
    await pointer(window, 'pointerup', { clientX: trackX(frame, 0.9) });
  });

  it('ignores the value region and secondary buttons when scrubbing', async () => {
    const { frame, onChange } = await mount();
    const button = valueButton();

    await pointer(frame, 'pointerdown', { button: 2, clientX: trackX(frame, 0.1) });

    if (button) {
      await pointer(button, 'pointerdown', { clientX: trackX(frame, 0.1) });
    }

    expect(onChange).not.toHaveBeenCalled();
    expect(frame.hasAttribute('data-dragging')).toBe(false);
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

  it('discards the draft on Escape and ignores unparseable input', async () => {
    const { onChange, slider } = await mount();

    await act(async () => {
      slider.focus();
      await userEvent.keyboard('{Enter}');
    });
    await act(() => userEvent.keyboard('77{Escape}'));

    expect(onChange).not.toHaveBeenCalled();
    expect(editor()).toBeNull();
    expect(document.activeElement).toBe(slider);

    await act(() => userEvent.keyboard('{Enter}'));
    await act(() => userEvent.keyboard('abc{Enter}'));

    expect(onChange).not.toHaveBeenCalled();
  });

  it('starts typing from a digit key and commits on blur, clamped to the typed range', async () => {
    const { onChange, slider } = await mount({ inputMax: 150 });

    await act(async () => {
      slider.focus();
      await userEvent.keyboard('9');
    });

    expect(editor()?.value).toBe('9');

    await act(() => userEvent.keyboard('99'));
    await act(() => {
      editor()?.blur();
    });

    expect(onChange).toHaveBeenLastCalledWith(150);
    expect(editor()).toBeNull();
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

  it('ignores pointers other than the one that started the drag', async () => {
    const { frame, onChange } = await mount();

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5), pointerId: 1 });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.8), pointerId: 2 });
    await pointer(window, 'pointerup', { clientX: trackX(frame, 0.8), pointerId: 2 });

    expect(onChange).not.toHaveBeenCalled();
    expect(frame.hasAttribute('data-dragging')).toBe(true);

    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.6), pointerId: 1 });
    await pointer(window, 'pointerup', { clientX: trackX(frame, 0.6), pointerId: 1 });

    expect(onChange).toHaveBeenLastCalledWith(40);
    expect(frame.hasAttribute('data-dragging')).toBe(false);
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

  it('marks stops as under, over, or at the value, adds the default, and skips the implied end stops', async () => {
    await mount({ defaultValue: 30, marks: [0, 10, 30, 60, 100], value: 30 });

    const states = [...(host?.querySelectorAll('[data-part="mark"]') ?? [])].map((mark) =>
      mark.getAttribute('data-state')
    );

    expect(states).toEqual(['over-value', 'at-value', 'under-value']);
  });

  it('keeps the slider range valid around a typed value beyond the track', async () => {
    const { slider } = await mount({ inputMax: 1000, value: 150 });

    expect(slider.getAttribute('aria-valuenow')).toBe('150');
    expect(slider.getAttribute('aria-valuemax')).toBe('150');
    const thumb = host?.querySelector<HTMLElement>('[data-part="thumb"]')?.getBoundingClientRect();
    const frame = host?.querySelector('[data-scope="scrubber"]')?.getBoundingClientRect();

    // Clamped to the track end rather than drawn past the frame (±1px for the frame border).
    expect(
      thumb && frame ? Math.abs(frame.right - (thumb.left + thumb.width / 2) - TRACK_INSET_PX) : NaN
    ).toBeLessThanOrEqual(1);
  });

  it('stays relative once Shift has been pressed mid-drag, in both directions', async () => {
    const { frame, onChange } = await mount();

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5) });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.6) });

    expect(onChange).toHaveBeenLastCalledWith(40);

    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.6), shiftKey: true });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.7), shiftKey: true });

    expect(onChange).toHaveBeenLastCalledWith(41);

    // Releasing Shift re-anchors at full sensitivity rather than jumping to the pointer.
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.7) });
    await pointer(window, 'pointermove', { clientX: trackX(frame, 0.8) });

    expect(onChange).toHaveBeenLastCalledWith(51);
    await pointer(window, 'pointercancel', {});

    expect(frame.hasAttribute('data-dragging')).toBe(false);
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

  it('hides the stops that would sit under the label or the value', async () => {
    // Label "Steps" spans roughly the first 15% of a 400px frame; the value the last ~10%.
    await mount({ marks: [3, 50, 97], value: 50 });
    await act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );

    const hidden = [...(host?.querySelectorAll('[data-part="mark"]') ?? [])].map((mark) =>
      mark.hasAttribute('data-hidden')
    );

    expect(hidden).toEqual([true, false, true]);
  });

  it('does nothing while disabled', async () => {
    const { frame, onChange, slider } = await mount({ defaultValue: 10, disabled: true });

    await pointer(frame, 'pointerdown', { clientX: trackX(frame, 0.5) });
    await act(() => frame.dispatchEvent(new MouseEvent('dblclick', { bubbles: true })));
    await act(async () => {
      slider.focus();
      await userEvent.keyboard('{ArrowRight}');
    });
    await act(() => {
      valueButton()?.click();
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(editor()).toBeNull();
    expect(slider.getAttribute('tabindex')).toBe('-1');
    expect(valueButton()?.disabled).toBe(true);
  });

  it('describes the slider with the error line and flags the frame invalid', async () => {
    const { frame, slider } = await mount({ error: 'Too high', helpText: 'Ignored while erroring' });
    const described = host?.querySelector(`#${CSS.escape(slider.getAttribute('aria-describedby') ?? '')}`);

    expect(described?.textContent).toBe('Too high');
    expect(described?.getAttribute('role')).toBe('alert');
    expect(frame.hasAttribute('data-invalid')).toBe(true);
  });
});
