import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { createDomRasterBackend } from '@workbench/canvas-engine/render/raster';
import { bboxHandleAt } from '@workbench/canvas-engine/tools/bboxHitTest';
import { transformOverlayGeometry } from '@workbench/canvas-engine/transform/transformMath';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it } from 'vitest';

import { createCheckerboardTile } from './compositor';
import { renderOverlay } from './overlayRenderer';

const CSS_SIZE = 200;
const RECT = { height: 50, width: 50, x: 50, y: 50 };
const IDENTITY_TRANSFORM = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

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
  it('draws the west bbox handle where pointer hit testing finds it', () => {
    for (const dpr of DPRS) {
      const { target, viewport } = renderAt(dpr);
      const columns = whiteColumnsAt(target, dpr);
      const centerCss = { x: (columns[0]! + columns.at(-1)! + 1) / 2 / dpr, y: 75 };

      expect(bboxHandleAt(transformBounds(viewport.viewMatrix(1), RECT), centerCss)).toBe('w');
      expect(Math.abs(centerCss.x - RECT.x)).toBeLessThanOrEqual(1);
    }
  });
});

describe('bbox frame over any content', () => {
  const FRAME_CSS_SIZE = 400;
  type Rgb = readonly number[];
  const ACCENT = [59, 130, 246] as const;
  const DPR_ZOOMS = [1, 1.25, 1.5, 2].flatMap((dpr) => [0.5, 1, 2.6].map((zoom) => [dpr, zoom] as const));

  const renderFrame = (dpr: number, zoom: number, background: Rgb, handles = false) => {
    const viewport = createViewport({ pan: { x: 7.3, y: 4.6 }, zoom });
    viewport.setViewportSize(FRAME_CSS_SIZE, FRAME_CSS_SIZE, dpr);
    const size = Math.round(FRAME_CSS_SIZE * dpr);
    const overlay = createDomRasterBackend().createSurface(size, size);
    renderOverlay(overlay, {
      bbox: RECT,
      bboxHandles: handles,
      dpr,
      view: viewport.viewMatrix(1),
      viewportSize: viewport.getViewportSize(),
    });
    const scene = createDomRasterBackend().createSurface(size, size);
    scene.ctx.fillStyle = `rgb(${background.join(', ')})`;
    scene.ctx.fillRect(0, 0, size, size);
    scene.ctx.drawImage(overlay.canvas, 0, 0);
    const read = (surface: typeof scene, x: number, y: number, w: number, h: number) => {
      const data = surface.ctx.getImageData(x, y, w, h).data;
      return Array.from({ length: w * h }, (_, i) => Array.from(data.subarray(i * 4, i * 4 + 4)));
    };
    return {
      column: (x: number, y: number, reach: number) => {
        const top = Math.floor((y - reach) * dpr);
        const height = Math.ceil(reach * 2 * dpr);
        const sx = Math.floor(x * dpr);
        return { overlay: read(overlay, sx, top, 1, height), scene: read(scene, sx, top, 1, height), start: top };
      },
      row: (x: number, y: number, reach: number) => {
        const left = Math.floor((x - reach) * dpr);
        const width = Math.ceil(reach * 2 * dpr);
        const sy = Math.floor(y * dpr);
        return { overlay: read(overlay, left, sy, width, 1), scene: read(scene, left, sy, width, 1), start: left };
      },
      screen: transformBounds(viewport.viewMatrix(1), RECT),
    };
  };

  const acrossTopEdge = (dpr: number, zoom: number, background: Rgb) => {
    const { column, screen } = renderFrame(dpr, zoom, background);
    return { ...column(screen.x + 2.75, screen.y, 4), edge: screen.y * dpr };
  };

  const isOpaque = (pixel: Rgb, color: Rgb): boolean =>
    pixel[3] === 255 && color.every((channel, index) => Math.abs(pixel[index]! - channel) <= 3);

  const coreOf = (overlay: readonly Rgb[]): number[] =>
    overlay.flatMap((pixel, index) => (isOpaque(pixel, ACCENT) ? [index] : []));

  it.each(DPR_ZOOMS)(
    'draws a crisp accent core centred where the edge falls, one hairline wide at any zoom (%sx, zoom %s)',
    (dpr, zoom) => {
      const { edge, overlay, start } = acrossTopEdge(dpr, zoom, ACCENT);
      const core = coreOf(overlay);

      expect(core).toHaveLength(Math.max(1, Math.round(dpr)));
      expect(core.at(-1)! - core[0]!).toBe(core.length - 1);
      const center = start + (core[0]! + core.at(-1)! + 1) / 2;
      expect(Math.abs(center - edge)).toBeLessThanOrEqual(0.5);
    }
  );
});

describe('color picker loupe', () => {
  const RED = [239, 18, 52, 255];
  const GREEN = [18, 200, 52, 255];
  const CHECKER = createCheckerboardTile(createDomRasterBackend(), { a: '#101010', b: '#202020' });

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

    expect(at(100, 100)).toEqual(GREEN);
    expect(at(110, 100)).toEqual(RED);
    expect(at(90, 90)).toEqual(RED);
    expect([
      [16, 16, 16, 255],
      [32, 32, 32, 255],
    ]).toContainEqual(at(46, 100));
    expect(at(100, 166)).toEqual([0, 0, 0, 0]);
  });
});
