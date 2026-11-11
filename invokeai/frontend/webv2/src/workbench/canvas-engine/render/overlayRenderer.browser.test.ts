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

describe('bbox frame over any content', () => {
  /** Large enough to hold the whole frame at every probed zoom. */
  const FRAME_CSS_SIZE = 400;
  /** An RGB colour, or an RGBA pixel whose alpha is ignored. */
  type Rgb = readonly number[];
  /** The frame's own accent: the background the frame used to vanish over. */
  const ACCENT = [59, 130, 246] as const;
  const OUTSIDE = [23, 37, 84] as const;
  const INSIDE = [255, 255, 255] as const;
  /** The accent, both extremes, and the image blues and mid tones between them that a single dark casing misses. */
  const BACKGROUNDS: readonly (readonly [string, Rgb])[] = [
    ['its accent', ACCENT],
    ['white', [255, 255, 255]],
    ['black', [0, 0, 0]],
    ['blue-600', [37, 99, 235]],
    ['blue-700', [29, 78, 216]],
    ['blue-800', [30, 64, 175]],
    ['blue-900', [30, 58, 138]],
    ['sky-700', [3, 105, 161]],
    ['mid grey', [102, 102, 102]],
    ['dark green', [47, 93, 58]],
  ];
  /** WCAG's minimum for a graphical object that identifies a UI component. */
  const MIN_CONTRAST = 3;
  const DPR_ZOOMS = [1, 1.25, 1.5, 2].flatMap((dpr) => [0.5, 1, 2.6].map((zoom) => [dpr, zoom] as const));

  const luminance = ([r = 0, g = 0, b = 0]: Rgb): number => {
    const linear = (channel: number) => {
      const c = channel / 255;
      return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
  };
  const contrast = (a: Rgb, b: Rgb): number => {
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (high! + 0.05) / (low! + 0.05);
  };

  /**
   * Renders the frame at `dpr` and `zoom` (a fractional pan keeps its edges off the pixel grid) and composites it over
   * an opaque `background`. Lines are read in device pixels: `overlay` alone, and `scene` as the user sees it.
   */
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
      /** Device pixels down the column through CSS `x`, from `y - reach` to `y + reach`, with their device rows. */
      column: (x: number, y: number, reach: number) => {
        const top = Math.floor((y - reach) * dpr);
        const height = Math.ceil(reach * 2 * dpr);
        const sx = Math.floor(x * dpr);
        return { overlay: read(overlay, sx, top, 1, height), scene: read(scene, sx, top, 1, height), start: top };
      },
      /** Device pixels along the row through CSS `y`, from `x - reach` to `x + reach`, with their device columns. */
      row: (x: number, y: number, reach: number) => {
        const left = Math.floor((x - reach) * dpr);
        const width = Math.ceil(reach * 2 * dpr);
        const sy = Math.floor(y * dpr);
        return { overlay: read(overlay, left, sy, width, 1), scene: read(scene, left, sy, width, 1), start: left };
      },
      screen: transformBounds(viewport.viewMatrix(1), RECT),
    };
  };

  /** Device rows across the top edge inside its first dash, which starts at the corner, clear of the left edge. */
  const acrossTopEdge = (dpr: number, zoom: number, background: Rgb) => {
    const { column, screen } = renderFrame(dpr, zoom, background);
    return { ...column(screen.x + 2.75, screen.y, 4), edge: screen.y * dpr };
  };

  const isOpaque = (pixel: Rgb, color: Rgb): boolean =>
    pixel[3] === 255 && color.every((channel, index) => Math.abs(pixel[index]! - channel) <= 3);

  /** Offsets (into the probe) of opaque accent pixels in the overlay: the frame's core line. */
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

  it.each(DPR_ZOOMS)('keeps the frame edge visible over blues, greys, white and black (%sx, zoom %s)', (dpr, zoom) => {
    for (const [name, background] of BACKGROUNDS) {
      const { overlay, scene } = acrossTopEdge(dpr, zoom, background);
      const core = coreOf(overlay);
      const contrastAt = (index: number) => contrast(scene[index]!, background);
      const coreContrast = Math.min(...core.map(contrastAt));
      const outside = Math.max(...scene.slice(0, core[0]).map((_, index) => contrastAt(index)));
      const inside = Math.max(...scene.slice(core.at(-1)! + 1).map((_, index) => contrastAt(core.at(-1)! + 1 + index)));

      // Some band of the line contrasts with any content; where the core matches it, both flanks still bound it.
      expect(core.length, name).toBeGreaterThan(0);
      expect(Math.max(coreContrast, outside, inside), name).toBeGreaterThanOrEqual(MIN_CONTRAST);
      if (background === ACCENT) {
        expect(Math.min(outside, inside), name).toBeGreaterThanOrEqual(MIN_CONTRAST);
      }
    }
  });

  it.each([1, 1.25, 1.5, 2])(
    'mirrors its bands on opposite edges of a pixel-aligned frame, only the white one inside it (%sx)',
    (dpr) => {
      // At zoom 0.8 RECT spans 40..80 CSS px: whole device pixels at every probed ratio.
      const viewport = createViewport({ pan: { x: 0, y: 0 }, zoom: 0.8 });
      viewport.setViewportSize(FRAME_CSS_SIZE, FRAME_CSS_SIZE, dpr);
      const size = Math.round(FRAME_CSS_SIZE * dpr);
      const overlay = createDomRasterBackend().createSurface(size, size);
      renderOverlay(overlay, {
        bbox: RECT,
        dpr,
        view: viewport.viewMatrix(1),
        viewportSize: viewport.getViewportSize(),
      });
      const screen = transformBounds(viewport.viewMatrix(1), RECT);
      const band = (x: number, y: number) => {
        const pixel = Array.from(overlay.ctx.getImageData(x, y, 1, 1).data);
        return isOpaque(pixel, OUTSIDE)
          ? 'navy'
          : isOpaque(pixel, ACCENT)
            ? 'core'
            : isOpaque(pixel, INSIDE)
              ? 'white'
              : pixel[3] === 0
                ? 'clear'
                : `partial ${pixel.join(',')}`;
      };
      const [left, top] = [screen.x * dpr, screen.y * dpr];
      const [right, bottom] = [(screen.x + screen.width) * dpr, (screen.y + screen.height) * dpr];
      // Inside each edge's first dash, which starts at a corner.
      const row = Math.floor((screen.y + 2.75) * dpr);
      const column = Math.floor((screen.x + 2.75) * dpr);
      // Device pixels from four outside each edge to four inside it, listed outside to inside.
      const offsets = [-4, -3, -2, -1, 0, 1, 2, 3];
      const leftBands = offsets.map((o) => band(left + o, row));

      expect(Number.isInteger(left) && Number.isInteger(bottom)).toBe(true);
      expect(offsets.map((o) => band(right - 1 - o, row))).toEqual(leftBands);
      expect(offsets.map((o) => band(column, top + o))).toEqual(leftBands);
      expect(offsets.map((o) => band(column, bottom - 1 - o))).toEqual(leftBands);
      if (Math.round(dpr) === 1) {
        // A one-pixel core sits on the pixel just outside; only the white band covers the area the frame marks.
        expect(leftBands).toEqual(['clear', 'clear', 'navy', 'core', 'white', 'clear', 'clear', 'clear']);
      } else {
        // A two-pixel core straddles the edge evenly.
        expect(leftBands).toEqual(['clear', 'navy', 'navy', 'core', 'core', 'white', 'white', 'clear']);
      }
    }
  );

  it.each(DPR_ZOOMS)('joins every drawn corner whole, with no notch at the dash seam (%sx, zoom %s)', (dpr, zoom) => {
    const viewport = createViewport({ pan: { x: 7.3, y: 4.6 }, zoom });
    viewport.setViewportSize(FRAME_CSS_SIZE, FRAME_CSS_SIZE, dpr);
    const size = Math.round(FRAME_CSS_SIZE * dpr);
    const overlay = createDomRasterBackend().createSurface(size, size);
    renderOverlay(overlay, { bbox: RECT, dpr, view: viewport.viewMatrix(1), viewportSize: viewport.getViewportSize() });
    const screen = transformBounds(viewport.viewMatrix(1), RECT);
    const hairline = Math.max(1, Math.round(dpr));
    // The core occupies the hairline's device pixels nearest each edge.
    const coreStart = (css: number) =>
      hairline % 2 === 1 ? Math.floor(css * dpr) - (hairline - 1) / 2 : Math.round(css * dpr) - hairline / 2;
    const block = (x: number, y: number) => {
      const data = overlay.ctx.getImageData(x, y, hairline, hairline).data;
      return Array.from({ length: hairline * hairline }, (_, i) => Array.from(data.subarray(i * 4, i * 4 + 4)));
    };
    const corners = [
      [screen.x, screen.y, -1, -1],
      [screen.x + screen.width, screen.y, 1, -1],
      [screen.x + screen.width, screen.y + screen.height, 1, 1],
      [screen.x, screen.y + screen.height, -1, 1],
    ] as const;
    let drawn = 0;
    for (const [cx, cy, outX, outY] of corners) {
      const x = coreStart(cx);
      const y = coreStart(cy);
      const reach = hairline * 2;
      const around = overlay.ctx.getImageData(x - reach, y - reach, hairline + reach * 2, hairline + reach * 2).data;
      // Coverage is whole: a seam or clipped join would leave partly covered pixels.
      for (let i = 3; i < around.length; i += 4) {
        expect(around[i]! <= 8 || around[i]! >= 247, `corner ${cx},${cy} alpha ${around[i]}`).toBe(true);
      }
      const outer = block(x + outX * hairline, y + outY * hairline);
      if (block(x, y).every((pixel) => isOpaque(pixel, ACCENT))) {
        drawn += 1;
        expect(
          outer.every((pixel) => isOpaque(pixel, OUTSIDE)),
          `corner ${cx},${cy} outside`
        ).toBe(true);
        const inner = block(x - outX * hairline, y - outY * hairline);
        expect(
          inner.every((pixel) => isOpaque(pixel, INSIDE)),
          `corner ${cx},${cy} inside`
        ).toBe(true);
      } else {
        expect(
          outer.every((pixel) => pixel[3]! <= 8),
          `corner ${cx},${cy} stray casing`
        ).toBe(true);
      }
    }
    // Every corner is a dash by construction, and the whole frame lies on the surface.
    expect(drawn).toBe(4);
  });

  // RECT spans 50..100; at 2x its edges land 0.3 px beyond each side of a 200 px surface, leaving only the white
  // inside band on the surface's outermost row or column.
  it.each([
    ['a deep-zoomed corner, edges running far off the surface', { a: 40, b: 0, c: 0, d: 40, e: -1950.3, f: -1950.6 }],
    ['the top edge just above the surface', { a: 2, b: 0, c: 0, d: 2, e: 7, f: -100.3 }],
    ['the bottom edge just below the surface', { a: 2, b: 0, c: 0, d: 2, e: 7, f: 0.3 }],
    ['the left edge just left of the surface', { a: 2, b: 0, c: 0, d: 2, e: -100.3, f: 7 }],
    ['the right edge just right of the surface', { a: 2, b: 0, c: 0, d: 2, e: 0.3, f: 7 }],
  ])('draws %s exactly as a larger surface shows it there', (_, view) => {
    const pad = 100;
    const render = (size: number, shift: number) => {
      const target = createDomRasterBackend().createSurface(size, size);
      renderOverlay(target, {
        bbox: RECT,
        dpr: 1,
        view: { ...view, e: view.e + shift, f: view.f + shift },
        viewportSize: { height: size, width: size },
      });
      return target.ctx.getImageData(shift, shift, 200, 200).data;
    };

    expect(render(200, 0)).toEqual(render(200 + pad * 2, pad));
  });

  it.each([1, 1.25, 1.5, 2])(
    'outlines the resize handles so they read over blues, greys, white and black (%sx)',
    (dpr) => {
      for (const [name, background] of BACKGROUNDS) {
        const { row, screen } = renderFrame(dpr, 1, background, true);
        // Through the west handle, away from the frame's dashes along its row.
        const center = { x: screen.x, y: screen.y + screen.height / 2 };
        const { scene, start } = row(center.x, center.y, 10);
        const visible = scene.flatMap((pixel, index) => (contrast(pixel, background) >= MIN_CONTRAST ? [index] : []));

        expect(visible.length, name).toBeGreaterThan(0);
        const widthCss = (visible.at(-1)! - visible[0]! + 1) / dpr;
        const centerCss = (start + (visible[0]! + visible.at(-1)! + 1) / 2) / dpr;
        // At least the white face, about 7 CSS px in whole device pixels.
        expect(widthCss, name).toBeGreaterThanOrEqual(6.5);
        expect(widthCss, name).toBeLessThanOrEqual(12);
        expect(Math.abs(centerCss - center.x), name).toBeLessThanOrEqual(1);
      }
    }
  );
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
