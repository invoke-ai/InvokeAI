import { describe, expect, it } from 'vitest';

import type { FloatingWidgetState, WidgetRegion, WidgetRegionState } from './layoutContracts';

import {
  clampWindowToViewport,
  commitResizedAxes,
  fitWindowIntoViewport,
  FLOATING_MIN_HEIGHT_PX,
  FLOATING_MIN_WIDTH_PX,
  getRegionOrder,
  normalizeFloatingPlacement,
  resizeFloatingGeometry,
  writeRegionOrder,
} from './floatingWindows';

const floating = (overrides: Partial<FloatingWidgetState> = {}): FloatingWidgetState => ({
  heightPx: 400,
  mode: 'windowed',
  returnRegion: 'right',
  stackOrder: 1,
  widthPx: 500,
  x: 40,
  y: 40,
  ...overrides,
});

const region = (instanceIds: string[], activeInstanceId = instanceIds[0] ?? ''): WidgetRegionState => ({
  activeInstanceId,
  instanceIds,
  isCollapsed: false,
  sizePx: 320,
});

const regions = (
  overrides: Partial<Record<WidgetRegion, WidgetRegionState>>
): Record<WidgetRegion, WidgetRegionState> => ({
  bottom: region([]),
  center: region(['canvas']),
  left: region([]),
  right: region([]),
  ...overrides,
});

const orderIds = (slots: { instanceId: string; isFloating: boolean }[]): string[] =>
  slots.map((slot) => (slot.isFloating ? `${slot.instanceId}*` : slot.instanceId));

describe('getRegionOrder', () => {
  it('places each marker at its index in the complete order, docked members filling the rest', () => {
    const slots = getRegionOrder('right', ['b', 'd'], {
      a: { returnIndex: 0, returnRegion: 'right' },
      c: { returnIndex: 2, returnRegion: 'right' },
      // Returns elsewhere, so it has no slot here.
      z: { returnIndex: 0, returnRegion: 'left' },
    });

    expect(orderIds(slots)).toEqual(['a*', 'b', 'c*', 'd']);
  });

  it('breaks colliding indices by instance id, whatever order the windows are stored or stacked in', () => {
    const forwards = getRegionOrder('right', ['x'], {
      b: { returnIndex: 1, returnRegion: 'right' },
      a: { returnIndex: 1, returnRegion: 'right' },
    });
    const backwards = getRegionOrder('right', ['x'], {
      a: { returnIndex: 1, returnRegion: 'right' },
      b: { returnIndex: 1, returnRegion: 'right' },
    });

    expect(orderIds(forwards)).toEqual(['x', 'a*', 'b*']);
    expect(backwards).toEqual(forwards);
  });

  it('appends markers whose index is missing, nonsensical, or past the end', () => {
    const slots = getRegionOrder('right', ['x'], {
      beyond: { returnIndex: 99, returnRegion: 'right' },
      missing: { returnRegion: 'right' },
      nan: { returnIndex: Number.NaN, returnRegion: 'right' },
      negative: { returnIndex: -1, returnRegion: 'right' },
    });

    // The finite index comes first; the rest append in id order.
    expect(orderIds(slots)).toEqual(['x', 'beyond*', 'missing*', 'nan*', 'negative*']);
  });
});

describe('writeRegionOrder', () => {
  it('round-trips an order it wrote, so docking one marker never moves another', () => {
    const floatingWidgets = { a: floating(), c: floating() };
    const written = writeRegionOrder(
      [
        { instanceId: 'a', isFloating: true },
        { instanceId: 'b', isFloating: false },
        { instanceId: 'c', isFloating: true },
        { instanceId: 'd', isFloating: false },
      ],
      floatingWidgets
    );

    expect(written.instanceIds).toEqual(['b', 'd']);
    expect(written.floatingWidgets).toMatchObject({ a: { returnIndex: 0 }, c: { returnIndex: 2 } });
    expect(orderIds(getRegionOrder('right', written.instanceIds, written.floatingWidgets))).toEqual([
      'a*',
      'b',
      'c*',
      'd',
    ]);
  });

  it('keeps the window map untouched when every marker already holds its index', () => {
    const floatingWidgets = { a: floating({ returnIndex: 1 }) };
    const written = writeRegionOrder(
      [
        { instanceId: 'b', isFloating: false },
        { instanceId: 'a', isFloating: true },
      ],
      floatingWidgets
    );

    expect(written.floatingWidgets).toBe(floatingWidgets);
  });
});

describe('normalizeFloatingPlacement', () => {
  const hasInstance = () => true;

  it('takes a floating instance out of every region and gives each marker its canonical index', () => {
    const normalized = normalizeFloatingPlacement(
      regions({
        // Stored before floating removed every membership: the window is still listed in two regions.
        center: region(['canvas', 'preview'], 'preview'),
        right: region(['layers', 'preview', 'queue'], 'preview'),
      }),
      { preview: floating({ returnIndex: 1 }), queue: floating() },
      hasInstance
    );

    expect(normalized.widgetRegions.center).toMatchObject({ activeInstanceId: 'canvas', instanceIds: ['canvas'] });
    expect(normalized.widgetRegions.right).toMatchObject({ activeInstanceId: 'layers', instanceIds: ['layers'] });
    expect(normalized.floatingWidgets).toMatchObject({ preview: { returnIndex: 1 }, queue: { returnIndex: 2 } });
  });

  it('leaves an emptied center naming the view it lost, and collapses an emptied rail', () => {
    const normalized = normalizeFloatingPlacement(
      regions({ center: region(['preview']), right: region(['preview']) }),
      { preview: floating() },
      hasInstance
    );

    expect(normalized.widgetRegions.center).toMatchObject({
      activeInstanceId: 'preview',
      instanceIds: [],
      isCollapsed: false,
    });
    expect(normalized.widgetRegions.right).toMatchObject({ activeInstanceId: '', instanceIds: [], isCollapsed: true });
  });

  it('changes nothing on a second pass', () => {
    const once = normalizeFloatingPlacement(
      regions({ center: region(['preview']), right: region(['x', 'preview', 'y'], 'preview') }),
      { preview: floating(), late: floating({ returnIndex: 7 }), early: floating({ returnIndex: 0 }) },
      hasInstance
    );
    const twice = normalizeFloatingPlacement(once.widgetRegions, once.floatingWidgets, hasInstance);

    expect(twice).toEqual(once);
  });

  it('drops windows it cannot place and reports none as undefined', () => {
    const widgetRegions = regions({ right: region(['x']) });
    const normalized = normalizeFloatingPlacement(
      widgetRegions,
      {
        gone: floating(),
        nowhere: floating({ returnRegion: 'nowhere' as WidgetRegion }),
        unsized: floating({ widthPx: Number.NaN }),
      },
      (instanceId) => instanceId !== 'gone'
    );

    expect(normalized.floatingWidgets).toBeUndefined();
    expect(normalized.widgetRegions).toBe(widgetRegions);
  });

  it('returns a window floated out of a retired right-rail dock to the rail', () => {
    const normalized = normalizeFloatingPlacement(
      regions({ right: region(['x']) }),
      { a: floating({ returnRegion: 'rightBottom' as WidgetRegion }) },
      hasInstance
    );

    expect(normalized.floatingWidgets?.a).toMatchObject({ returnIndex: 1, returnRegion: 'right' });
  });
});

describe('resizeFloatingGeometry', () => {
  const start = { heightPx: 400, widthPx: 500, x: 100, y: 80 };

  it('moves only the dragged side, keeping the opposite one anchored', () => {
    expect(resizeFloatingGeometry(start, 'e', 40, 999)).toEqual({ ...start, widthPx: 540 });
    expect(resizeFloatingGeometry(start, 's', 999, 30)).toEqual({ ...start, heightPx: 430 });
    expect(resizeFloatingGeometry(start, 'w', -40, 999)).toEqual({ ...start, widthPx: 540, x: 60 });
    expect(resizeFloatingGeometry(start, 'n', 999, -30)).toEqual({ ...start, heightPx: 430, y: 50 });
    expect(resizeFloatingGeometry(start, 'nw', 10, 20)).toEqual({ heightPx: 380, widthPx: 490, x: 110, y: 100 });
  });

  it('stops at the minimum size without letting the anchored side drift', () => {
    const shrunk = resizeFloatingGeometry(start, 'nw', 5000, 5000);

    expect(shrunk).toEqual({
      heightPx: FLOATING_MIN_HEIGHT_PX,
      widthPx: FLOATING_MIN_WIDTH_PX,
      x: 100 + 500 - FLOATING_MIN_WIDTH_PX,
      y: 80 + 400 - FLOATING_MIN_HEIGHT_PX,
    });
    expect(resizeFloatingGeometry(start, 'se', -5000, -5000)).toEqual({
      ...start,
      heightPx: FLOATING_MIN_HEIGHT_PX,
      widthPx: FLOATING_MIN_WIDTH_PX,
    });
  });

  it('stops the top edge at the top of the viewport, growing only as far as that', () => {
    expect(resizeFloatingGeometry(start, 'n', 0, -500)).toEqual({ ...start, heightPx: 480, y: 0 });
  });
});

describe('resizeFloatingGeometry within a viewport', () => {
  const viewport = { heightPx: 500, widthPx: 600 };

  it('does not grow past the viewport, and keeps the anchored side where it is when it stops', () => {
    const start = { heightPx: 300, widthPx: 600, x: 0, y: 40 };

    // Already as wide as the viewport: dragging the left edge further left changes nothing.
    expect(resizeFloatingGeometry(start, 'w', -80, 0, viewport)).toEqual(start);
    expect(resizeFloatingGeometry(start, 'e', 80, 0, viewport)).toEqual(start);
    expect(resizeFloatingGeometry(start, 's', 0, 900, viewport)).toEqual({ ...start, heightPx: 500 });
  });

  it('never raises an axis to a minimum the viewport is smaller than', () => {
    const narrow = { heightPx: 500, widthPx: 240 };
    const start = { heightPx: 300, widthPx: 240, x: 0, y: 40 };

    expect(resizeFloatingGeometry(start, 'se', 0, 20, narrow)).toEqual({ ...start, heightPx: 320 });
    expect(resizeFloatingGeometry(start, 'e', -60, 0, narrow)).toEqual(start);
  });
});

describe('commitResizedAxes', () => {
  // Stored for a larger display; shown clamped and capped by a 600x500 viewport.
  const stored = { heightPx: 900, widthPx: 900, x: 5000, y: 40 };
  const start = { heightPx: 500, widthPx: 600, x: 552, y: 40 };

  it('takes the resized values only on the axes the resize changed', () => {
    expect(commitResizedAxes(stored, start, { ...start, widthPx: 550 })).toEqual({ ...stored, widthPx: 550, x: 552 });
    expect(commitResizedAxes(stored, start, { ...start, heightPx: 350, y: 60 })).toEqual({
      ...stored,
      heightPx: 350,
      y: 60,
    });
    expect(commitResizedAxes(stored, start, { heightPx: 350, widthPx: 550, x: 560, y: 60 })).toEqual({
      heightPx: 350,
      widthPx: 550,
      x: 560,
      y: 60,
    });
  });

  it('commits nothing when the resize changed nothing, as a drag against the cap or the minimum does', () => {
    expect(commitResizedAxes(stored, start, { ...start })).toBeNull();
  });

  it('does not read sub-pixel arithmetic on an axis as a resize of it', () => {
    // At a fractional zoom the rectangle on screen is not on whole pixels, and the untouched axis comes back a
    // fraction off: it must keep what was stored, not take the viewport's cap.
    expect(commitResizedAxes(stored, start, { ...start, heightPx: 350, widthPx: 600.3 })).toEqual({
      ...stored,
      heightPx: 350,
    });
    expect(commitResizedAxes(stored, start, { ...start, widthPx: 599.7, x: 552.2 })).toBeNull();
  });
});

describe('fitWindowIntoViewport', () => {
  it('brings a rectangle remembered from a larger viewport wholly on screen, keeping its size', () => {
    const remembered = { heightPx: 360, widthPx: 480, x: 700, y: 520 };

    expect(fitWindowIntoViewport(remembered, { height: 450, width: 720 })).toEqual({ ...remembered, x: 240, y: 90 });
    // Larger than the viewport: pinned to the corner; CSS caps what is shown and the size is kept.
    expect(fitWindowIntoViewport({ ...remembered, widthPx: 2000 }, { height: 450, width: 720 })).toMatchObject({
      widthPx: 2000,
      x: 0,
    });
    expect(fitWindowIntoViewport({ ...remembered, x: -300, y: -20 }, { height: 900, width: 1440 })).toMatchObject({
      x: 0,
      y: 0,
    });
    // Already on screen: untouched.
    expect(fitWindowIntoViewport(remembered, { height: 900, width: 1440 })).toEqual(remembered);
  });
});

describe('clampWindowToViewport', () => {
  it('keeps a sliver of the width that is shown, not of a wider stored one', () => {
    // Shown no wider than the 600px viewport, so the left limit is 48 - 600, not 48 - 2000.
    const clamped = clampWindowToViewport(
      { heightPx: 300, widthPx: 2000, x: -1900, y: 20 },
      { height: 500, width: 600 }
    );

    expect(clamped.x).toBe(48 - 600);
  });
});
