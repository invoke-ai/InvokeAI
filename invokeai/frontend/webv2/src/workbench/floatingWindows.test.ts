import { describe, expect, it } from 'vitest';

import type { FloatingWidgetState, WidgetRegion, WidgetRegionState } from './layoutContracts';

import { getRegionOrder, normalizeFloatingPlacement, writeRegionOrder } from './floatingWindows';

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
