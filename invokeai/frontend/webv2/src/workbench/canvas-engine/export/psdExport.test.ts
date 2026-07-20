import type { CanvasBlendMode } from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import type { PsdExportLayerInput } from './psdExport';

import { blendModeToPsd, planPsdExport, PSD_MAX_DIMENSION } from './psdExport';

const IDENTITY = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

const layer = (over: Partial<PsdExportLayerInput> = {}): PsdExportLayerInput => ({
  blendMode: 'normal',
  contentRect: { height: 50, width: 100, x: 0, y: 0 },
  id: 'a',
  isEnabled: true,
  name: 'Layer',
  opacity: 1,
  transform: { ...IDENTITY },
  ...over,
});

const findLayer = (plan: ReturnType<typeof planPsdExport>, id: string) => {
  if (plan.status !== 'ok') {
    throw new Error(`expected ok plan, got ${plan.status}`);
  }
  const found = plan.layers.find((l) => l.id === id);
  if (!found) {
    throw new Error(`layer ${id} not in plan`);
  }
  return found;
};

describe('blendModeToPsd', () => {
  it('maps every canvas blend mode to a PSD blend key', () => {
    const cases: [CanvasBlendMode, string][] = [
      ['normal', 'normal'],
      ['multiply', 'multiply'],
      ['screen', 'screen'],
      ['overlay', 'overlay'],
      ['darken', 'darken'],
      ['lighten', 'lighten'],
      ['color-dodge', 'color dodge'],
      ['color-burn', 'color burn'],
      ['hard-light', 'hard light'],
      ['soft-light', 'soft light'],
      ['difference', 'difference'],
      ['exclusion', 'exclusion'],
      ['hue', 'hue'],
      ['saturation', 'saturation'],
      ['color', 'color'],
      ['luminosity', 'luminosity'],
    ];
    for (const [mode, key] of cases) {
      expect(blendModeToPsd(mode)).toBe(key);
    }
  });

  it('falls back to normal for an unknown blend mode', () => {
    expect(blendModeToPsd('made-up' as CanvasBlendMode)).toBe('normal');
  });
});

describe('planPsdExport', () => {
  it('returns empty when there are no layers', () => {
    expect(planPsdExport([])).toEqual({ status: 'empty' });
  });

  it('returns empty when every layer has empty content', () => {
    expect(planPsdExport([layer({ contentRect: { height: 0, width: 0, x: 0, y: 0 } })])).toEqual({
      status: 'empty',
    });
  });

  it('sizes the PSD canvas to a single layer content bounds', () => {
    const plan = planPsdExport([layer()]);
    expect(plan.status).toBe('ok');
    if (plan.status !== 'ok') {
      return;
    }
    expect(plan.width).toBe(100);
    expect(plan.height).toBe(50);
    expect(plan.canvasRect).toEqual<Rect>({ height: 50, width: 100, x: 0, y: 0 });
    const only = plan.layers[0]!;
    expect(only).toMatchObject({ bottom: 50, hidden: false, left: 0, opacity: 1, right: 100, top: 0 });
  });

  it('unions world-space bounds across layers and positions each relative to the origin', () => {
    const plan = planPsdExport([
      layer({ id: 'top', transform: { ...IDENTITY, x: 50, y: 20 } }),
      layer({ id: 'bottom', transform: { ...IDENTITY, x: -30, y: -10 } }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    // union of [-30,-10,100,50] and [50,20,100,50] = [-30,-10, 180, 80]
    expect(plan.canvasRect).toEqual<Rect>({ height: 80, width: 180, x: -30, y: -10 });
    expect(plan.width).toBe(180);
    expect(plan.height).toBe(80);
    // positions are relative to the union origin (-30, -10)
    expect(findLayer(plan, 'bottom')).toMatchObject({ left: 0, top: 0, right: 100, bottom: 50 });
    expect(findLayer(plan, 'top')).toMatchObject({ left: 80, top: 30, right: 180, bottom: 80 });
  });

  it('emits layers bottom-to-top (input is top-first, PSD children are bottom-first)', () => {
    const plan = planPsdExport([layer({ id: 'top' }), layer({ id: 'mid' }), layer({ id: 'bottom' })]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    expect(plan.layers.map((l) => l.id)).toEqual(['bottom', 'mid', 'top']);
  });

  it('marks disabled layers hidden but still exports them, and clamps opacity to 0..1', () => {
    const plan = planPsdExport([
      layer({ id: 'shown', isEnabled: true, opacity: 0.5 }),
      layer({ id: 'over', isEnabled: true, opacity: 2 }),
      layer({ id: 'hidden', isEnabled: false, opacity: -1 }),
    ]);
    expect(findLayer(plan, 'shown')).toMatchObject({ hidden: false, opacity: 0.5 });
    expect(findLayer(plan, 'over')).toMatchObject({ opacity: 1 });
    expect(findLayer(plan, 'hidden')).toMatchObject({ hidden: true, opacity: 0 });
  });

  it('maps blend modes and reports unmapped ones (falling back to normal)', () => {
    const plan = planPsdExport([
      layer({ blendMode: 'multiply', id: 'a' }),
      layer({ blendMode: 'nonsense' as CanvasBlendMode, id: 'b' }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    expect(findLayer(plan, 'a').blendMode).toBe('multiply');
    expect(findLayer(plan, 'b').blendMode).toBe('normal');
    expect(plan.unmappedBlends).toEqual(['nonsense']);
  });

  it('passes non-destructive adjustments through for the executor to bake', () => {
    const adjustments = { brightness: 0.2, contrast: 0, saturation: 0 };
    const plan = planPsdExport([layer({ adjustments })]);
    expect(findLayer(plan, 'a').adjustments).toBe(adjustments);
  });

  it('drops empty-content layers from the plan but keeps the rest', () => {
    const plan = planPsdExport([
      layer({ contentRect: { height: 0, width: 0, x: 0, y: 0 }, id: 'empty' }),
      layer({ id: 'real' }),
    ]);
    if (plan.status !== 'ok') {
      throw new Error('expected ok');
    }
    expect(plan.layers.map((l) => l.id)).toEqual(['real']);
  });

  it('refuses an export whose union bounds exceed the dimension cap', () => {
    const plan = planPsdExport([layer({ contentRect: { height: 10, width: 100, x: 0, y: 0 } })], {
      maxDimension: 50,
    });
    expect(plan).toEqual({ height: 10, status: 'too-large', width: 100 });
  });

  it('accepts bounds exactly at the cap', () => {
    const plan = planPsdExport([layer({ contentRect: { height: 10, width: PSD_MAX_DIMENSION, x: 0, y: 0 } })]);
    expect(plan.status).toBe('ok');
  });
});
