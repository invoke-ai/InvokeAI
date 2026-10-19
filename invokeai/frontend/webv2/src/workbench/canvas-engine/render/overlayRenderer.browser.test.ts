import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { bboxHandleAt } from '@workbench/canvas-engine/tools/bboxHitTest';
import { transformOverlayGeometry, transformTargetAt } from '@workbench/canvas-engine/transform/transformMath';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it } from 'vitest';

import { createCheckerboardTile } from './compositor';
import { colorLoupePixels, renderOverlay } from './overlayRenderer';

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

describe('color picker loupe', () => {
  const RED = [239, 18, 52, 255];
  const GREEN = [18, 200, 52, 255];
  const CHECKER = createCheckerboardTile(createDomRasterBackend(), { a: '#101010', b: '#202020' });

  /** A square of `size` pixels: red, transparent on the left two columns, green at the sampled center. */
  const loupePixels = (size: number) => {
    const pixels = createDomRasterBackend().createSurface(size, size);
    pixels.ctx.fillStyle = 'rgb(239, 18, 52)';
    pixels.ctx.fillRect(2, 0, size - 2, size);
    pixels.ctx.fillStyle = 'rgb(18, 200, 52)';
    pixels.ctx.fillRect((size - 1) / 2, (size - 1) / 2, 1, 1);
    return pixels;
  };

  const renderLoupe = (dpr: number, color: { r: number; g: number; b: number; a: number } | null, size = 15) => {
    const viewport = createViewport();
    viewport.setViewportSize(CSS_SIZE, CSS_SIZE, dpr);
    const target = createDomRasterBackend().createSurface(
      Math.round(CSS_SIZE * viewport.getDpr()),
      Math.round(CSS_SIZE * viewport.getDpr())
    );
    renderOverlay(target, {
      bbox: { height: 0, width: 0, x: -1000, y: -1000 },
      colorLoupe: {
        checker: CHECKER,
        color,
        pixels: loupePixels(size),
        point: viewport.screenToDocument({ x: 100, y: 100 }),
      },
      dpr: viewport.getDpr(),
      showBbox: false,
      view: viewport.viewMatrix(1),
      viewportSize: viewport.getViewportSize(),
    });
    return (x: number, y: number) => [...target.ctx.getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data];
  };

  it.each([1, 2])('magnifies the pixels around the pointer without smoothing at %sx', (dpr) => {
    const at = renderLoupe(dpr, null);

    // The sampled pixel fills the 8 px cell under the pointer; its neighbours fill the next cells.
    expect(at(100, 100)).toEqual(GREEN);
    expect(at(110, 100)).toEqual(RED);
    expect(at(90, 90)).toEqual(RED);
    // Empty pixels show the checker, and nothing is drawn beyond the 60 px ring.
    expect([
      [16, 16, 16, 255],
      [32, 32, 32, 255],
    ]).toContainEqual(at(46, 100));
    expect(at(100, 166)).toEqual([0, 0, 0, 0]);
  });

  it('shows fewer, larger pixels as the canvas zooms past the loupe', () => {
    expect([0.25, 1, 4, 8, 20].map(colorLoupePixels)).toEqual([15, 15, 15, 7, 3]);
    const at = renderLoupe(1, null, 7);

    // Seven pixels span 120 px, so the sampled one covers about 17 px around the pointer (boxed at its edge).
    expect(at(105, 100)).toEqual(GREEN);
    expect(at(112, 100)).toEqual(RED);
  });

  it('labels the sampled color below the center, and shows no label over empty canvas', () => {
    expect(renderLoupe(1, null)(100, 125)).toEqual(RED);
    expect(renderLoupe(1, { a: 255, b: 52, g: 200, r: 18 })(100, 125)).not.toEqual(RED);
  });
});
