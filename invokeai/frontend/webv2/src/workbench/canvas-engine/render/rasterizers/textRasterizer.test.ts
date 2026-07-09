import type { RasterCallLogEntry, StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';

import { createLayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it } from 'vitest';

import type { TextSource } from './textRasterizer';
import type { RasterizeDeps } from './types';

import { estimateTextExtent, rasterizeTextSource, TEXT_CHAR_WIDTH_FACTOR, textFontString } from './textRasterizer';

const makeDeps = (): RasterizeDeps => {
  const backend = createTestStubRasterBackend();
  return {
    backend,
    documentSize: { height: 200, width: 300 },
    resolver: () => Promise.resolve(new Blob()),
    store: createLayerCacheStore(backend),
  };
};

const ops = (surface: StubRasterSurface): string[] => surface.callLog.map((e) => e.op);
const entries = (surface: StubRasterSurface, op: string): RasterCallLogEntry[] =>
  surface.callLog.filter((e) => e.op === op);

const text = (over: Partial<TextSource> = {}): TextSource => ({
  align: 'left',
  color: '#112233',
  content: 'hello',
  fontFamily: 'Inter',
  fontSize: 20,
  fontWeight: 400,
  lineHeight: 1.5,
  type: 'text',
  ...over,
});

// The stub's deterministic metric: width = chars × fontSizePx × TEXT_CHAR_WIDTH_FACTOR.
const stubWidth = (chars: number, fontSize: number): number => Math.ceil(chars * fontSize * TEXT_CHAR_WIDTH_FACTOR);

describe('rasterizeTextSource — measurement + extent', () => {
  it('sizes the surface to the measured block (widest line × fontSize×0.6, lines × fontSize×lineHeight)', async () => {
    const deps = makeDeps();
    const source = text({ content: 'hello', fontSize: 20, lineHeight: 1.5 });
    const surface = (await rasterizeTextSource(source, deps)).surface as StubRasterSurface;
    expect(surface.width).toBe(stubWidth(5, 20)); // 'hello' = 5 chars → 60
    expect(surface.height).toBe(Math.ceil(1 * 20 * 1.5)); // one line → 30
  });

  it('agrees with the pure estimateTextExtent (cache-size stability)', async () => {
    const deps = makeDeps();
    const source = text({ content: 'abc\ndefgh', fontSize: 16, lineHeight: 1.2 });
    const surface = (await rasterizeTextSource(source, deps)).surface as StubRasterSurface;
    const estimate = estimateTextExtent(source);
    expect(surface.width).toBe(estimate.width);
    expect(surface.height).toBe(estimate.height);
  });

  it('clears before drawing and applies the font + fill color', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeTextSource(text({ color: '#abcdef' }), deps)).surface as StubRasterSurface;
    expect(ops(surface)).toContain('clearRect');
    expect(
      surface.callLog.some((e) => e.op === 'set' && e.args[0] === 'font' && e.args[1] === textFontString(text()))
    ).toBe(true);
    expect(surface.callLog.some((e) => e.op === 'set' && e.args[0] === 'fillStyle' && e.args[1] === '#abcdef')).toBe(
      true
    );
  });
});

describe('rasterizeTextSource — line breaks', () => {
  it('draws one fillText per manual line, stepped by fontSize×lineHeight', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeTextSource(text({ content: 'a\nbb\nccc', fontSize: 10, lineHeight: 2 }), deps))
      .surface as StubRasterSurface;
    const fills = entries(surface, 'fillText');
    expect(fills).toHaveLength(3);
    // step = 10 × 2 = 20; y anchors at 0, 20, 40.
    expect(fills.map((e) => e.args[2])).toEqual([0, 20, 40]);
    expect(fills.map((e) => e.args[0])).toEqual(['a', 'bb', 'ccc']);
    // Height covers all three lines.
    expect(surface.height).toBe(60);
  });
});

describe('rasterizeTextSource — alignment', () => {
  it('left-aligns at x=0', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeTextSource(text({ align: 'left' }), deps)).surface as StubRasterSurface;
    expect(surface.callLog.some((e) => e.op === 'set' && e.args[0] === 'textAlign' && e.args[1] === 'left')).toBe(true);
    expect(entries(surface, 'fillText')[0]?.args[1]).toBe(0);
  });

  it('center-aligns at x=width/2', async () => {
    const deps = makeDeps();
    const source = text({ align: 'center', content: 'hello', fontSize: 20 });
    const surface = (await rasterizeTextSource(source, deps)).surface as StubRasterSurface;
    expect(surface.callLog.some((e) => e.op === 'set' && e.args[0] === 'textAlign' && e.args[1] === 'center')).toBe(
      true
    );
    expect(entries(surface, 'fillText')[0]?.args[1]).toBe(stubWidth(5, 20) / 2);
  });

  it('right-aligns at x=width', async () => {
    const deps = makeDeps();
    const source = text({ align: 'right', content: 'hello', fontSize: 20 });
    const surface = (await rasterizeTextSource(source, deps)).surface as StubRasterSurface;
    expect(surface.callLog.some((e) => e.op === 'set' && e.args[0] === 'textAlign' && e.args[1] === 'right')).toBe(
      true
    );
    expect(entries(surface, 'fillText')[0]?.args[1]).toBe(stubWidth(5, 20));
  });
});

describe('rasterizeTextSource — empty + reuse', () => {
  it('produces a minimal 1×lineHeight surface for empty content', async () => {
    const deps = makeDeps();
    const surface = (await rasterizeTextSource(text({ content: '', fontSize: 20, lineHeight: 1.5 }), deps))
      .surface as StubRasterSurface;
    expect(surface.width).toBe(1);
    expect(surface.height).toBe(30);
  });

  it('resizes and reuses a provided target surface', async () => {
    const deps = makeDeps();
    const target = deps.backend.createSurface(5, 5);
    const source = text({ content: 'hello', fontSize: 20 });
    const { surface } = await rasterizeTextSource(source, deps, target);
    expect(surface).toBe(target);
    expect(surface.width).toBe(stubWidth(5, 20));
  });
});
