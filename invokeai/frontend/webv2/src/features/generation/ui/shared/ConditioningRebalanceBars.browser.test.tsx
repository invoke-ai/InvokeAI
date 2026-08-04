import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ConditioningRebalanceBars } from './ConditioningRebalanceBars';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onActiveIndexChange = vi.fn();
const onCommit = vi.fn();
const onPreview = vi.fn();

const TAP_COUNT = 12;
/** All weights at 1.0, so `getRebalanceBarScale` reports the nominal ceiling. */
const FLAT_WEIGHTS = Array.from({ length: TAP_COUNT }, () => 1);
const SCALE = 8;

const tapLabel = (tap: number, layer: number) => `Tap ${tap}, encoder layer ${layer}`;

const renderBars = async (weights: readonly number[] = FLAT_WEIGHTS) => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ConditioningRebalanceBars
          activeIndex={null}
          tapLabel={tapLabel}
          weights={weights}
          onActiveIndexChange={onActiveIndexChange}
          onCommit={onCommit}
          onPreview={onPreview}
        />
      </ChakraProvider>
    )
  );

  const bars = [...document.querySelectorAll('[role="slider"]')];
  const track = bars[0]?.parentElement;

  if (bars.length !== TAP_COUNT || !track) {
    throw new Error('rebalance bars did not render');
  }

  return { bars, rect: track.getBoundingClientRect(), track };
};

const interact = async (run: () => void) => {
  await act(async () => {
    run();
    await Promise.resolve();
  });
};

/** Pointer coordinates that land on `index`'s column at `weight` on the track. */
const pointAt = (rect: DOMRect, index: number, weight: number) => ({
  clientX: rect.left + (index + 0.5) * (rect.width / TAP_COUNT),
  clientY: rect.top + (1 - weight / SCALE) * rect.height,
});

const lastPreview = (): number[] => (onPreview.mock.calls.at(-1)?.[0] as number[] | null) ?? [];

const lastCommit = (): number[] => (onCommit.mock.calls.at(-1)?.[0] as number[] | undefined) ?? [];

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  onActiveIndexChange.mockClear();
  onCommit.mockClear();
  onPreview.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ConditioningRebalanceBars', () => {
  it('names each bar by its tap and the encoder layer it weights', async () => {
    const { bars } = await renderBars();

    expect(bars).toHaveLength(TAP_COUNT);
    expect(bars[0]?.getAttribute('aria-label')).toBe('Tap 1, encoder layer 2');
    expect(bars[7]?.getAttribute('aria-label')).toBe('Tap 8, encoder layer 23');
    expect(bars[11]?.getAttribute('aria-label')).toBe('Tap 12, encoder layer 35');
  });

  it('exposes the weight range to assistive tech', async () => {
    const { bars } = await renderBars();
    const bar = bars[0];

    expect(bar?.getAttribute('aria-orientation')).toBe('vertical');
    expect(bar?.getAttribute('aria-valuemin')).toBe('0');
    expect(bar?.getAttribute('aria-valuemax')).toBe(String(SCALE));
    expect(bar?.getAttribute('aria-valuenow')).toBe('1');
    expect(bar?.getAttribute('tabindex')).toBe('0');
  });

  it('rescales rather than clipping when a weight overshoots the nominal ceiling', async () => {
    const { bars } = await renderBars([...FLAT_WEIGHTS.slice(0, 11), 12.5]);

    expect(bars[0]?.getAttribute('aria-valuemax')).toBe('12.5');
  });

  it('drags one bar to the pointed-at weight and commits once on release', async () => {
    const { rect, track } = await renderBars();
    const target = pointAt(rect, 3, 6);

    await interact(() => {
      track.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, ...target }));
    });

    expect(lastPreview()[3]).toBeCloseTo(6, 1);
    // Neighbours are untouched by a stationary press.
    expect(lastPreview()[2]).toBe(1);
    expect(lastPreview()[4]).toBe(1);
    expect(onCommit).not.toHaveBeenCalled();

    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', target)));

    expect(onPreview).toHaveBeenLastCalledWith(null);
    expect(onCommit).toHaveBeenCalledOnce();
    expect(lastCommit()[3]).toBeCloseTo(6, 1);
  });

  it('paints across every column a sweep passes, including ones it skipped', async () => {
    const { rect, track } = await renderBars();

    await interact(() => {
      track.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, ...pointAt(rect, 0, 0) }));
    });
    // One large move from the first column to the last: the columns in between get
    // no pointer sample of their own and must still be filled in.
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', pointAt(rect, 11, 8))));

    const painted = lastPreview();

    expect(painted[0]).toBeCloseTo(0, 1);
    expect(painted[11]).toBeCloseTo(8, 1);

    for (let index = 1; index < TAP_COUNT; index += 1) {
      expect(painted[index] ?? 0).toBeGreaterThan(painted[index - 1] ?? 0);
    }
  });

  it('clamps a drag that leaves the track instead of reporting an impossible weight', async () => {
    const { rect, track } = await renderBars();
    // Stay in column 5's x band throughout, so this measures vertical clamping only.
    const { clientX } = pointAt(rect, 5, 4);

    await interact(() => {
      track.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, ...pointAt(rect, 5, 4) }));
    });

    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX, clientY: -9000 })));
    expect(lastPreview()[5]).toBe(SCALE);

    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX, clientY: 9000 })));
    expect(lastPreview()[5]).toBe(0);
  });

  it('adjusts a bar by keyboard on the step grid', async () => {
    const { bars } = await renderBars();
    const bar = bars[4];

    await interact(() => bar?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowUp' })));
    expect(lastCommit()[4]).toBeCloseTo(1.1, 5);

    await interact(() =>
      bar?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown', shiftKey: true }))
    );
    expect(lastCommit()[4]).toBe(0);

    await interact(() => bar?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'End' })));
    expect(lastCommit()[4]).toBe(SCALE);

    await interact(() => bar?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Home' })));
    expect(lastCommit()[4]).toBe(0);
  });

  it('leaves the other taps alone when one is adjusted by keyboard', async () => {
    const { bars } = await renderBars();

    await interact(() => bars[4]?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowUp' })));

    const committed = lastCommit();

    expect(committed.filter((weight) => weight !== 1)).toHaveLength(1);
  });

  it('moves focus between bars with the horizontal arrows', async () => {
    const { bars } = await renderBars();

    await act(() => {
      (bars[4] as HTMLElement).focus();
    });
    await interact(() => bars[4]?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));

    expect(document.activeElement).toBe(bars[5]);
    expect(onCommit).not.toHaveBeenCalled();

    await interact(() => bars[5]?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' })));

    expect(document.activeElement).toBe(bars[4]);
  });

  it('reports the tap under the pointer so the parent can read it out', async () => {
    const { rect, track } = await renderBars();

    await interact(() => {
      track.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, ...pointAt(rect, 7, 3) }));
    });

    expect(onActiveIndexChange).toHaveBeenLastCalledWith(7);
  });
});
