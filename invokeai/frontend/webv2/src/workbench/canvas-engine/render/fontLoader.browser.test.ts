import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';

import { createFontRuntime, type FontRecord } from '@features/fonts';
import { afterEach, describe, expect, it } from 'vitest';

import customAxisWideFontUrl from './fixtures/custom-axis-wide.ttf?url&inline';
import customAxisFontUrl from './fixtures/custom-axis.ttf?url&inline';
import { createFontLoader } from './fontLoader';
import { createLayerCacheStore } from './layerCache';
import { createDomRasterBackend } from './raster';
import { rasterizeTextSource } from './rasterizers/textRasterizer';

type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;

const loaders: Array<{ dispose: () => void }> = [];

const sourceFor = (overrides: Partial<TextSource> = {}): TextSource => ({
  align: 'left',
  color: '#fff',
  content: 'A',
  fontFamily: 'Invoke Font Fixture',
  fontRef: {
    contentHash: 'a'.repeat(64),
    family: 'Invoke Font Fixture',
    id: 'browser-variable',
    label: 'Invoke Font Fixture Regular',
  },
  fontSize: 64,
  fontWeight: 400,
  lineHeight: 1.2,
  type: 'text',
  ...overrides,
});

const glyphPixels = (source: Awaited<ReturnType<typeof rasterizeTextSource>>): number => {
  const pixels = source.surface.ctx.getImageData(0, 0, source.surface.width, source.surface.height).data;
  let count = 0;
  for (let index = 3; index < pixels.length; index += 4) {
    if (pixels[index] > 0) {
      count += 1;
    }
  }
  return count;
};

afterEach(() => {
  for (const loader of loaders.splice(0)) {
    loader.dispose();
  }
});

describe('Canvas custom-font loading in Chromium', () => {
  it('applies requested bold styling to a static regular face', async () => {
    const runtime = createFontRuntime({
      download: async () => new Uint8Array(await (await fetch(customAxisWideFontUrl)).arrayBuffer()),
      getFont: (id): Promise<FontRecord> =>
        Promise.resolve({
          id,
          contentHash: 'a'.repeat(64),
          family: 'Invoke Font Fixture',
          label: 'Regular',
          filename: 'regular.ttf',
          axes: [],
          instances: [],
          byteSize: 1,
          scope: 'private',
          source: 'uploaded',
          style: 'normal',
          weight: 400,
          url: '',
        }),
    });
    const loader = createFontLoader(runtime);
    loaders.push({
      dispose: () => {
        loader.dispose();
        runtime.dispose();
      },
    });
    const normal = sourceFor();
    const bold = sourceFor({ fontWeight: 700 });
    loader.setActiveSources([normal, bold]);
    await Promise.all([loader.waitForReady(normal), loader.waitForReady(bold)]);
    const backend = createDomRasterBackend();
    const rasterize = (source: TextSource) =>
      rasterizeTextSource(source, {
        backend,
        documentSize: { height: 128, width: 128 },
        resolver: () => Promise.resolve(new Blob()),
        resolveFontFamily: loader.resolveFamily,
        store: createLayerCacheStore(backend),
      });
    const normalRaster = await rasterize(normal);
    const boldRaster = await rasterize(bold);
    expect(glyphPixels(boldRaster)).toBeGreaterThan(glyphPixels(normalRaster));
  });

  it('registers real variable and pinned static faces, then rasterizes their distinct glyph metrics', async () => {
    const variable = sourceFor({ fontVariations: { TEST: 0, wght: 400 } });
    const pinnedStatic = sourceFor({
      fontRef: {
        contentHash: 'b'.repeat(64),
        family: 'Invoke Font Fixture',
        id: 'browser-static-wide',
        label: 'Invoke Font Fixture Wide',
      },
      fontVariations: { TEST: 100, wght: 400 },
    });
    const runtime = createFontRuntime({
      download: async (reference) => {
        const url = reference.axes?.TEST === 100 ? customAxisWideFontUrl : customAxisFontUrl;
        return new Uint8Array(await (await fetch(url)).arrayBuffer());
      },
      getFont: (id): Promise<FontRecord> =>
        Promise.resolve({
          axes:
            id === 'browser-variable'
              ? [
                  { default: 400, hidden: false, label: 'Weight', maximum: 900, minimum: 100, tag: 'wght' },
                  { default: 0, hidden: false, label: 'Test', maximum: 100, minimum: 0, tag: 'TEST' },
                ]
              : [],
          byteSize: 1,
          contentHash: id === 'browser-static-wide' ? 'b'.repeat(64) : 'a'.repeat(64),
          family: 'Invoke Font Fixture',
          filename: `${id}.ttf`,
          id,
          instances: [],
          label: id === 'browser-static-wide' ? 'Invoke Font Fixture Wide' : 'Invoke Font Fixture Regular',
          scope: 'private',
          source: 'uploaded',
          style: 'normal',
          url: '',
          weight: 400,
        }),
    });
    const loader = createFontLoader(runtime);
    loaders.push({
      dispose: () => {
        loader.dispose();
        runtime.dispose();
      },
    });
    loader.setActiveSources([variable, pinnedStatic]);

    const [variableFamily, staticFamily] = await Promise.all([
      loader.waitForReady(variable),
      loader.waitForReady(pinnedStatic),
    ]);
    expect(variableFamily).toMatch(/^__invoke_font_/);
    expect(staticFamily).toMatch(/^__invoke_font_/);
    expect(staticFamily).not.toBe(variableFamily);
    expect(document.fonts.check(`400 64px ${variableFamily}`)).toBe(true);
    expect(document.fonts.check(`400 64px ${staticFamily}`)).toBe(true);

    const backend = createDomRasterBackend();
    const variableRaster = await rasterizeTextSource(variable, {
      backend,
      documentSize: { height: 128, width: 128 },
      resolver: () => Promise.resolve(new Blob()),
      resolveFontFamily: loader.resolveFamily,
      store: createLayerCacheStore(backend),
    });
    const staticRaster = await rasterizeTextSource(pinnedStatic, {
      backend,
      documentSize: { height: 128, width: 128 },
      resolver: () => Promise.resolve(new Blob()),
      resolveFontFamily: loader.resolveFamily,
      store: createLayerCacheStore(backend),
    });

    expect(glyphPixels(variableRaster)).toBeGreaterThan(0);
    expect(glyphPixels(staticRaster)).toBeGreaterThan(0);
    expect(staticRaster.rect.width).toBeGreaterThan(variableRaster.rect.width);
  });
});
