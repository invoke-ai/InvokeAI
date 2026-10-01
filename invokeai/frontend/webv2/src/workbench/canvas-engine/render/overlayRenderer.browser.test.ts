import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { bboxHandleAt } from '@workbench/canvas-engine/tools/bboxHitTest';
import { transformOverlayGeometry, transformTargetAt } from '@workbench/canvas-engine/transform/transformMath';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it } from 'vitest';

import { renderOverlay } from './overlayRenderer';

const CSS_SIZE = 200;
const RECT = { height: 50, width: 50, x: 50, y: 50 };
const IDENTITY_TRANSFORM = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

/** Renders bbox and transform chrome at `dpr` the way the engine does: CSS view and size, DPR base. */
const renderAt = (dpr: number) => {
  const viewport = createViewport();
  viewport.setViewportSize(CSS_SIZE, CSS_SIZE, dpr);
  const target = createDomRasterBackend().createSurface(
    Math.round(CSS_SIZE * viewport.getDpr()),
    Math.round(CSS_SIZE * viewport.getDpr())
  );
  renderOverlay(target, {
    bbox: RECT,
    bboxHandles: true,
    dpr: viewport.getDpr(),
    transformFrame: transformOverlayGeometry(IDENTITY_TRANSFORM, RECT),
    view: viewport.viewMatrix(1),
    viewportSize: viewport.getViewportSize(),
  });
  return { target, viewport };
};

const DPRS = [0.75, 1, 1.5, 2];

/** Backing columns of opaque white across one row through the west bbox handle. */
const whiteColumnsAt = (target: ReturnType<typeof renderAt>['target'], dpr: number): number[] => {
  const left = Math.round(40 * dpr);
  const row = target.ctx.getImageData(left, Math.round(75 * dpr), Math.round(20 * dpr), 1).data;
  const columns: number[] = [];
  for (let i = 0; i < row.length; i += 4) {
    if (row[i] === 255 && row[i + 1] === 255 && row[i + 2] === 255 && row[i + 3] === 255) {
      columns.push(left + i / 4);
    }
  }
  return columns;
};

describe('overlay chrome across device-pixel ratios', () => {
  it('keeps handle sizes constant in CSS pixels across ratios', () => {
    const cssWidths = DPRS.map((dpr) => whiteColumnsAt(renderAt(dpr).target, dpr).length / dpr);

    expect(cssWidths[0]).toBeGreaterThan(0);
    for (const width of cssWidths) {
      expect(Math.abs(width - cssWidths[0]!)).toBeLessThanOrEqual(1);
    }
  });

  it('draws the west bbox handle where pointer hit testing finds it', () => {
    for (const dpr of DPRS) {
      const { target, viewport } = renderAt(dpr);
      const columns = whiteColumnsAt(target, dpr);
      const centerCss = { x: (columns[0]! + columns.at(-1)! + 1) / 2 / dpr, y: 75 };

      expect(bboxHandleAt(transformBounds(viewport.viewMatrix(1), RECT), centerCss)).toBe('w');
      expect(Math.abs(centerCss.x - RECT.x)).toBeLessThanOrEqual(1);
    }
  });

  it('draws the rotation knob where pointer hit testing finds the rotate target', () => {
    for (const dpr of DPRS) {
      const { target, viewport } = renderAt(dpr);
      // The knob is the only white fill between the canvas top and the north scale handle (top edge at 46 CSS px).
      const x = Math.round(75 * dpr);
      const column = target.ctx.getImageData(x, 0, 1, Math.floor(44 * dpr)).data;
      const whiteRows: number[] = [];
      for (let row = 0; row < column.length / 4; row += 1) {
        if (column[row * 4] === 255 && column[row * 4 + 1] === 255 && column[row * 4 + 3] === 255) {
          whiteRows.push(row);
        }
      }
      expect(whiteRows.length).toBeGreaterThan(0);
      const knobCss = { x: 75, y: (whiteRows[0]! + whiteRows.at(-1)!) / 2 / dpr };

      expect(
        transformTargetAt({
          point: knobCss,
          rect: RECT,
          toScreen: (point) => viewport.documentToScreen(point),
          transform: IDENTITY_TRANSFORM,
        })
      ).toEqual({ handle: 'n', kind: 'rotate' });
      expect(Math.abs(knobCss.y - (50 - 18))).toBeLessThanOrEqual(1);
    }
  });

  it('sizes the backing store by a fractional or sub-one ratio and maps the view through the same ratio', () => {
    for (const dpr of [0.75, 1.5]) {
      const { target, viewport } = renderAt(dpr);
      expect(viewport.getDpr()).toBe(dpr);
      expect(target.width).toBe(Math.round(CSS_SIZE * dpr));
      expect(viewport.viewMatrix(viewport.getDpr()).a).toBe(dpr);
    }
  });
});
