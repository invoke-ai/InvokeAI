import type { CanvasControlLayerContract } from '@workbench/canvas-engine/contracts';

import { createControlLayer, needsControlDetection } from '@workbench/canvas-engine/api';
import { describe, expect, it } from 'vitest';

import { applyControlLayerIntent } from './controlLayerIntents';

const image = (imageName: string, size = 64) => ({ height: size, imageName, width: size });
const photo = { image: image('photo.png'), type: 'image' } as const;
const mapSource = { bitmap: image('map.png'), offset: { x: 0, y: 0 }, type: 'paint' } as const;
const depth = { settings: { model_size: 'small_v2' }, type: 'depth_anything_depth_estimation' };
const options = {
  detectorDefaults: (detector: string): Record<string, unknown> =>
    ({
      canny_edge_detection: { high_threshold: 200, low_threshold: 100 },
      depth_anything_depth_estimation: { model_size: 'small_v2' },
      dw_openpose_detection: { draw_body: true },
      hed_edge_detection: { scribble: false },
      normal_map: {},
      pidi_edge_detection: { quantize_edges: false, scribble: false },
    })[detector] ?? {},
};
const SHIFTED = { rotation: 0, scaleX: 1, scaleY: 1, x: 40, y: 8 };

const layer = (patch: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  ...createControlLayer({ id: 'c1', name: 'Control Layer 1' }),
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: 'explicit-model',
    weight: 1,
  },
  ...patch,
});
/** A depth layer whose photo was detected into a map. */
const detected = (patch: Partial<CanvasControlLayerContract> = {}) =>
  layer({ controlType: 'depth', detection: depth, filter: depth, original: photo, source: mapSource, ...patch });

const apply = (target: CanvasControlLayerContract, intent: Parameters<typeof applyControlLayerIntent>[1]) => {
  const next = applyControlLayerIntent(target, intent, options);
  if (next === 'no-detection-input') {
    throw new Error('unexpected refusal');
  }
  return next;
};

describe('applyControlLayerIntent: set-type', () => {
  it('types an image with the default detector and returns to automatic model choice', () => {
    const next = apply(layer({ source: photo }), { controlType: 'depth', kind: 'set-type' });

    expect(next.controlType).toBe('depth');
    expect(next.detection).toEqual(depth);
    expect(next.adapter.model).toBeNull();
    expect(next.source).toEqual(photo);
    expect(needsControlDetection(next)).toBe(true);
  });

  it('chooses None for an empty layer and for a map without an original', () => {
    expect(apply(layer(), { controlType: 'depth', kind: 'set-type' }).detection).toBeNull();
    const legacyMap = layer({ filter: depth, source: mapSource });
    const typed = apply(legacyMap, { controlType: 'depth', kind: 'set-type' });

    expect(typed.detection).toBeNull();
    expect(typed.source).toEqual(mapSource);
  });

  it('keeps a detector the new type also offers, settings included', () => {
    const custom = { settings: { quantize_edges: true, scribble: true }, type: 'pidi_edge_detection' };
    const next = apply(layer({ controlType: 'softedge', detection: custom, source: photo }), {
      controlType: 'sketch',
      kind: 'set-type',
    });

    expect(next.detection).toBe(custom);
  });

  it("switches to the new type's default and keeps the original for re-detection", () => {
    const next = apply(detected(), { controlType: 'pose', kind: 'set-type' });

    expect(next.detection).toEqual({ settings: { draw_body: true }, type: 'dw_openpose_detection' });
    expect(next.original).toEqual(photo);
    expect(needsControlDetection(next)).toBe(true);
  });

  it('restores the original for a type without a default detector', () => {
    const next = apply(detected({ originalTransform: SHIFTED }), { controlType: 'tile', kind: 'set-type' });

    expect(next).toMatchObject({ controlType: 'tile', detection: null, source: photo });
    expect(next.transform).toMatchObject({ x: 40, y: 8 });
    expect(next.original).toBeUndefined();
    expect(next.filter).toBeUndefined();
  });

  it('renames in the same patch and reports an unchanged type as the same object', () => {
    const typed = detected();

    expect(apply(typed, { controlType: 'depth', kind: 'set-type' })).toBe(typed);
    const renamed = apply(typed, { controlType: 'depth', kind: 'set-type', name: 'Depth 1' });
    expect(renamed.name).toBe('Depth 1');
    expect(renamed.adapter.model).toBe('explicit-model');
    expect(apply(layer({ source: photo }), { controlType: 'depth', kind: 'set-type', name: 'Depth 1' }).name).toBe(
      'Depth 1'
    );
  });
});

describe('applyControlLayerIntent: set-detector', () => {
  it("starts from the detector's defaults merged with the type's fixed settings", () => {
    const next = apply(layer({ controlType: 'sketch', detection: null, source: photo }), {
      detector: 'pidi_edge_detection',
      kind: 'set-detector',
    });

    expect(next.detection).toEqual({
      settings: { quantize_edges: false, scribble: true },
      type: 'pidi_edge_detection',
    });
  });

  it('keeps the original so the new detector runs on it', () => {
    const next = apply(detected(), { detector: 'normal_map', kind: 'set-detector' });

    expect(next.detection).toEqual({ settings: {}, type: 'normal_map' });
    expect(next.original).toEqual(photo);
    expect(needsControlDetection(next)).toBe(true);
  });

  it('None restores the original', () => {
    const next = apply(detected(), { detector: null, kind: 'set-detector' });

    expect(next).toMatchObject({ detection: null, source: photo });
    expect(next.original).toBeUndefined();
  });

  it('refuses a detector for a map without an original', () => {
    expect(
      applyControlLayerIntent(
        layer({ controlType: 'depth', detection: null, filter: depth, source: mapSource }),
        { detector: 'normal_map', kind: 'set-detector' },
        options
      )
    ).toBe('no-detection-input');
  });

  it('leaves the same detector and an untyped layer unchanged', () => {
    const typed = detected();
    const untyped = layer({ source: photo });

    expect(apply(typed, { detector: depth.type, kind: 'set-detector' })).toBe(typed);
    expect(apply(untyped, { detector: depth.type, kind: 'set-detector' })).toBe(untyped);
  });
});

describe('applyControlLayerIntent: set-detector-settings', () => {
  it('changes only the settings and keeps fixed ones', () => {
    const sketch = layer({
      controlType: 'sketch',
      detection: { settings: { quantize_edges: false, scribble: true }, type: 'pidi_edge_detection' },
      source: photo,
    });
    const next = apply(sketch, { kind: 'set-detector-settings', settings: { quantize_edges: true, scribble: false } });

    expect(next).toEqual({
      ...sketch,
      detection: { settings: { quantize_edges: true, scribble: true }, type: 'pidi_edge_detection' },
    });
  });

  it('leaves a layer without a detector, or with equal settings, unchanged', () => {
    const none = layer({ controlType: 'depth', detection: null, source: photo });
    const typed = detected();

    expect(apply(none, { kind: 'set-detector-settings', settings: { a: 1 } })).toBe(none);
    expect(apply(typed, { kind: 'set-detector-settings', settings: { model_size: 'small_v2' } })).toBe(typed);
  });
});

describe('applyControlLayerIntent: replace-image and draw', () => {
  const upload = image('upload.png', 128);

  it('replaces the source, drops derived pixels and stales the map', () => {
    const next = apply(detected({ originalTransform: SHIFTED }), {
      image: upload,
      kind: 'replace-image',
      transform: SHIFTED,
    });

    expect(next.source).toEqual({ image: upload, type: 'image' });
    expect(next.transform).toEqual(SHIFTED);
    for (const key of ['original', 'originalTransform', 'filter'] as const) {
      expect(next).not.toHaveProperty(key);
    }
    expect(next.detection).toEqual(depth);
    expect(needsControlDetection(next)).toBe(true);
  });

  it("gives a first image the type's default detector when None was chosen only for lack of pixels", () => {
    const next = apply(layer({ controlType: 'depth', detection: null }), { image: upload, kind: 'replace-image' });

    expect(next.detection).toEqual(depth);
  });

  it('keeps None when the layer already had an image, or the type has no default', () => {
    expect(
      apply(layer({ controlType: 'depth', detection: null, source: photo }), { image: upload, kind: 'replace-image' })
        .detection
    ).toBeNull();
    expect(
      apply(layer({ controlType: 'sketch', detection: null }), { image: upload, kind: 'replace-image' }).detection
    ).toBeNull();
    expect(apply(layer(), { image: upload, kind: 'replace-image' })).not.toHaveProperty('detection');
  });

  it('draw empties the image and uses it as-is', () => {
    const next = apply(detected(), { kind: 'draw' });

    expect(next.source).toEqual({ bitmap: null, type: 'paint' });
    expect(next.detection).toBeNull();
    for (const key of ['original', 'originalTransform', 'filter'] as const) {
      expect(next).not.toHaveProperty(key);
    }
    expect(apply(layer({ source: photo }), { kind: 'draw' })).not.toHaveProperty('detection');
  });
});
