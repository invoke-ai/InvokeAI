import { describe, expect, it } from 'vitest';

import type { CanvasProjectState } from './canvasProjectFile';
import {
  CANVAS_PROJECT_VERSION,
  collectImageNames,
  isV2Manifest,
  parseManifest,
  remapCanvasState,
  remapRefImages,
} from './canvasProjectFile';

const baseBbox = {
  rect: { x: 0, y: 0, width: 512, height: 512 },
  modelBase: 'sd-1' as const,
  optimalDimension: 512,
  aspectRatio: { value: 1, isLocked: false, id: 'Free' as const },
  scaledSize: { width: 512, height: 512 },
  scaleMethod: 'auto' as const,
};

const makeEmptyState = (): CanvasProjectState => ({
  rasterLayers: [],
  controlLayers: [],
  inpaintMasks: [],
  regionalGuidance: [],
  bbox: baseBbox,
  selectedEntityIdentifier: null,
  bookmarkedEntityIdentifier: null,
});

/**
 * Builds a raster-layer entity carrying one image object. Used to seed states with image refs
 * so we can verify both `collectImageNames` and `remapCanvasState` walk the right fields.
 */
const makeRasterLayerWithImage = (layerId: string, imageName: string) =>
  ({
    id: layerId,
    name: null,
    type: 'raster_layer',
    isEnabled: true,
    isLocked: false,
    position: { x: 0, y: 0 },
    opacity: 1,
    objects: [
      {
        id: `${layerId}-img`,
        type: 'image',
        image: { image_name: imageName, width: 512, height: 512 },
      },
    ],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }) as any;

describe('parseManifest', () => {
  it('accepts a v2 manifest verbatim', () => {
    const manifest = {
      version: 2,
      appVersion: '6.14.0',
      createdAt: '2026-05-14T18:00:00.000Z',
      name: 'My Project',
      width: 1024,
      height: 1024,
      imageCount: 3,
      hasPreview: true,
    };
    const parsed = parseManifest(manifest);
    expect(parsed).toEqual(manifest);
    expect(isV2Manifest(parsed)).toBe(true);
  });

  it('accepts a v1 manifest (legacy format) without width/height/imageCount/hasPreview', () => {
    const manifest = {
      version: 1,
      appVersion: '6.11.1.post1',
      createdAt: '2026-02-26T19:09:01.440Z',
      name: 'Old Project',
    };
    const parsed = parseManifest(manifest);
    expect(parsed.version).toBe(1);
    expect(parsed.name).toBe('Old Project');
    expect(isV2Manifest(parsed)).toBe(false);
  });

  it('fills missing v1 name with empty string (legacy files predate the name field)', () => {
    const manifest = {
      version: 1,
      appVersion: '6.11.0',
      createdAt: '2026-01-01T00:00:00.000Z',
    };
    const parsed = parseManifest(manifest);
    expect(parsed.name).toBe('');
  });

  it('rejects an unknown version', () => {
    expect(() =>
      parseManifest({
        version: 99,
        appVersion: '?',
        createdAt: '?',
        name: '?',
      })
    ).toThrow();
  });

  it('rejects a v2 manifest with missing required fields', () => {
    expect(() =>
      parseManifest({
        version: 2,
        appVersion: '6.14.0',
        createdAt: '2026-05-14T18:00:00.000Z',
        name: 'Missing dims',
        // missing width/height/imageCount/hasPreview
      })
    ).toThrow();
  });

  it('declares CANVAS_PROJECT_VERSION as 2 so new saves are always v2', () => {
    expect(CANVAS_PROJECT_VERSION).toBe(2);
  });
});

describe('collectImageNames', () => {
  it('returns an empty set for an empty canvas + empty ref images', () => {
    expect(collectImageNames(makeEmptyState(), [])).toEqual(new Set());
  });

  it('walks raster layer image objects', () => {
    const state = makeEmptyState();
    state.rasterLayers = [makeRasterLayerWithImage('rl-1', 'a.png'), makeRasterLayerWithImage('rl-2', 'b.png')];
    expect(collectImageNames(state, [])).toEqual(new Set(['a.png', 'b.png']));
  });

  it('deduplicates names referenced by multiple layers', () => {
    const state = makeEmptyState();
    state.rasterLayers = [makeRasterLayerWithImage('rl-1', 'shared.png'), makeRasterLayerWithImage('rl-2', 'shared.png')];
    expect(collectImageNames(state, [])).toEqual(new Set(['shared.png']));
  });

  it('walks global ref images including their crop variant', () => {
    const refs = [
      {
        id: 'r1',
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        config: {
          image: {
            original: { image: { image_name: 'ref-orig.png', width: 1, height: 1 }, rect: { x: 0, y: 0, width: 1, height: 1 } },
            crop: { image: { image_name: 'ref-crop.png', width: 1, height: 1 }, rect: { x: 0, y: 0, width: 1, height: 1 } },
          },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } as any,
      },
    ];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(collectImageNames(makeEmptyState(), refs as any)).toEqual(new Set(['ref-orig.png', 'ref-crop.png']));
  });
});

describe('remapCanvasState', () => {
  it('returns the same state untouched when the mapping is empty', () => {
    const state = makeEmptyState();
    state.rasterLayers = [makeRasterLayerWithImage('rl-1', 'a.png')];
    const remapped = remapCanvasState(state, new Map());
    expect(remapped).toBe(state);
  });

  it('rewrites image_name in raster layer image objects', () => {
    const state = makeEmptyState();
    state.rasterLayers = [makeRasterLayerWithImage('rl-1', 'old.png')];
    const mapping = new Map([['old.png', 'new.png']]);
    const remapped = remapCanvasState(state, mapping);
    const firstObject = remapped.rasterLayers[0]?.objects[0];
    expect(firstObject?.type).toBe('image');
    if (firstObject?.type === 'image' && 'image_name' in firstObject.image) {
      expect(firstObject.image.image_name).toBe('new.png');
    }
  });

  it('leaves unmapped names alone', () => {
    const state = makeEmptyState();
    state.rasterLayers = [
      makeRasterLayerWithImage('rl-1', 'mapped.png'),
      makeRasterLayerWithImage('rl-2', 'unmapped.png'),
    ];
    const mapping = new Map([['mapped.png', 'remapped.png']]);
    const remapped = remapCanvasState(state, mapping);
    // After remap, collectImageNames should see the new name for `rl-1` and the original for `rl-2`.
    expect(collectImageNames(remapped, [])).toEqual(new Set(['remapped.png', 'unmapped.png']));
  });

  it('does not mutate the input state', () => {
    const state = makeEmptyState();
    state.rasterLayers = [makeRasterLayerWithImage('rl-1', 'old.png')];
    remapCanvasState(state, new Map([['old.png', 'new.png']]));
    // Inputs should be untouched — a fresh collect on the original still sees `old.png`.
    expect(collectImageNames(state, [])).toEqual(new Set(['old.png']));
  });
});

describe('remapRefImages', () => {
  it('returns the same array untouched when the mapping is empty', () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const refs: any[] = [];
    expect(remapRefImages(refs, new Map())).toBe(refs);
  });

  it('rewrites both original and crop image_name', () => {
    const refs = [
      {
        id: 'r1',
        config: {
          image: {
            original: { image: { image_name: 'orig.png', width: 1, height: 1 }, rect: { x: 0, y: 0, width: 1, height: 1 } },
            crop: { image: { image_name: 'crop.png', width: 1, height: 1 }, rect: { x: 0, y: 0, width: 1, height: 1 } },
          },
        },
      },
    ];
    const mapping = new Map([
      ['orig.png', 'orig-new.png'],
      ['crop.png', 'crop-new.png'],
    ]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const remapped = remapRefImages(refs as any, mapping);
    expect(remapped[0]?.config.image?.original.image.image_name).toBe('orig-new.png');
    expect(remapped[0]?.config.image?.crop?.image.image_name).toBe('crop-new.png');
  });
});
