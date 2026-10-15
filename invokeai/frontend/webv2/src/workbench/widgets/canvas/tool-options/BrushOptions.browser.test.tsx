import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/api';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { BRUSH_SIZE_TRACK_MAX, PaintPercentControl, PaintSizeControl } from './BrushOptions';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('paint percent control', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  afterEach(async () => {
    await act(() => root?.unmount());
    container?.remove();
    container = null;
    root = null;
  });

  it('edits a 0–1 option in whole percent, writing every step, and resets to the tool default', async () => {
    const setValue = vi.fn();
    const Harness = () => {
      const [value, setHarnessValue] = useState(0.5);
      const handleValue = useCallback((next: number) => {
        setValue(next);
        setHarnessValue(next);
      }, []);
      return <PaintPercentControl defaultValue={1} label="Opacity" setValue={handleValue} value={value} />;
    };
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const slider = page.getByRole('slider', { name: 'Opacity' });
    await expect.element(slider).toHaveAttribute('aria-valuetext', '50%');
    await act(async () => {
      (slider.element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight}{ArrowRight}');
    });
    // Tool options are not history: each step reaches the store, not just the settled value.
    expect(setValue.mock.calls).toEqual([[0.51], [0.52]]);
    await expect.element(slider).toHaveAttribute('aria-valuetext', '52%');

    await act(() => userEvent.keyboard('{Backspace}'));
    expect(setValue).toHaveBeenLastCalledWith(1);
    await expect.element(slider).toHaveAttribute('aria-valuetext', '100%');
  });
});

describe('paint size control', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  afterEach(async () => {
    await act(() => root?.unmount());
    container?.remove();
    container = null;
    root = null;
  });

  /** Owns the size like the tool store does, so each key steps from what the previous one produced. */
  const renderSize = async (initial: number, setSize = vi.fn(), controlled = true) => {
    const Harness = () => {
      const [size, setHarnessSize] = useState(initial);
      const handleSize = useCallback((next: number) => {
        setSize(next);
        if (controlled) {
          setHarnessSize(next);
        }
      }, []);
      return <PaintSizeControl label="Brush size" setSize={handleSize} size={size} />;
    };
    container = document.createElement('div');
    container.style.width = '300px';
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });
    return { setSize, slider: page.getByRole('slider', { name: 'Brush size' }) };
  };

  const typeSize = async (slider: ReturnType<typeof page.getByRole>, text: string) => {
    await act(async () => {
      (slider.element() as HTMLElement).focus();
      await userEvent.keyboard('{Enter}');
    });
    const input = page.getByRole('textbox', { name: 'Brush size' });
    await act(async () => {
      await userEvent.clear(input);
      if (text) {
        await userEvent.fill(input, text);
      }
      await userEvent.keyboard('{Enter}');
    });
  };

  /** Where the thumb sits along the track, 0–1 (the track is inset 10px from the frame). */
  const thumbFraction = () => {
    const frame = container!.querySelector<HTMLElement>('[data-scope="scrubber"]')!.getBoundingClientRect();
    const thumb = container!.querySelector<HTMLElement>('[data-part="thumb"]')!.getBoundingClientRect();
    return (thumb.left + thumb.width / 2 - frame.left - 10) / (frame.width - 20);
  };

  it('announces the actual size, including sub-pixel sizes', async () => {
    const { slider } = await renderSize(0.25);

    await expect.element(slider).toHaveAttribute('aria-valuetext', '0.25px');
    await expect.element(slider).toHaveAttribute('aria-valuenow', '0.25');
  });

  it('spreads sizes logarithmically so 0.1px–1px gets real travel and 600px ends the track', async () => {
    await renderSize(MIN_BRUSH_SIZE);
    expect(thumbFraction()).toBeCloseTo(0, 2);

    await act(() => root?.unmount());
    container?.remove();
    await renderSize(1);
    // A tenth of the 0.1px–600px ratio range, not the 0.15% a linear track would give it.
    expect(thumbFraction()).toBeCloseTo(Math.log(10) / Math.log(BRUSH_SIZE_TRACK_MAX / MIN_BRUSH_SIZE), 2);

    await act(() => root?.unmount());
    container?.remove();
    await renderSize(MAX_BRUSH_SIZE);
    expect(thumbFraction()).toBeCloseTo(1, 2);
  });

  it('advances from the minimum with the keyboard in sub-pixel steps', async () => {
    const { slider } = await renderSize(MIN_BRUSH_SIZE);

    await act(async () => {
      (slider.element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight}');
    });

    await expect.element(slider).toHaveAttribute('aria-valuetext', '0.11px');
  });

  it('supports PageUp and PageDown in logical brush units', async () => {
    const { slider } = await renderSize(MIN_BRUSH_SIZE);

    await act(async () => {
      (slider.element() as HTMLElement).focus();
      await userEvent.keyboard('{PageUp}');
    });
    await expect.element(slider).toHaveAttribute('aria-valuetext', '0.2px');

    await act(() => userEvent.keyboard('{PageDown}'));
    await expect.element(slider).toHaveAttribute('aria-valuetext', '0.1px');
  });

  it('reaches the exact track maximum with End, then keys keep stepping in 10px units like `]`', async () => {
    const { slider } = await renderSize(50);

    await act(async () => {
      (slider.element() as HTMLElement).focus();
      await userEvent.keyboard('{End}');
    });
    await expect.element(slider).toHaveAttribute('aria-valuetext', '600px');

    await act(() => userEvent.keyboard('{ArrowRight}{PageUp}'));
    await expect.element(slider).toHaveAttribute('aria-valuetext', '710px');
  });

  it('steps from the actual size above the track instead of jumping back onto it', async () => {
    const { slider } = await renderSize(1000);

    await act(async () => {
      (slider.element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight}');
    });
    await expect.element(slider).toHaveAttribute('aria-valuetext', '1010px');

    await act(() => userEvent.keyboard('{ArrowLeft}{ArrowLeft}'));
    await expect.element(slider).toHaveAttribute('aria-valuetext', '990px');
  });

  it('lets a decimal be typed before committing the size, up to the brush limit', async () => {
    const { setSize, slider } = await renderSize(1);

    await typeSize(slider, '0.25');
    expect(setSize).toHaveBeenLastCalledWith(0.25);
    await expect.element(slider).toHaveAttribute('aria-valuetext', '0.25px');

    await typeSize(slider, '5000');
    expect(setSize).toHaveBeenLastCalledWith(MAX_BRUSH_SIZE);
  });

  it('canonicalizes a valid over-precision commit even when the engine value is unchanged', async () => {
    const { setSize, slider } = await renderSize(0.25, vi.fn(), false);

    await typeSize(slider, '0.254');

    expect(setSize).toHaveBeenLastCalledWith(0.25);
    await expect.element(slider).toHaveAttribute('aria-valuetext', '0.25px');
  });

  it('keeps the current size when an empty edit is committed', async () => {
    const { setSize, slider } = await renderSize(5);

    await typeSize(slider, '');

    expect(setSize).not.toHaveBeenCalled();
    await expect.element(slider).toHaveAttribute('aria-valuetext', '5px');
  });
});
