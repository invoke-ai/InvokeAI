import type { CanvasControlLayerContract } from '@workbench/canvas-engine/contracts';

import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { layerTransformMatrix } from '@workbench/canvas-engine/transform/transformMath';
import { describe, expect, it } from 'vitest';

import {
  controlDetectionInputKey,
  detachControlDerivation,
  getControlDetectionInput,
  isControlMapCurrent,
  needsControlDetection,
  rebaseControlOriginal,
  restoreControlOriginal,
  withDetectedMap,
} from './controlDerivation';
import { createControlLayer } from './layerFactories';

const image = (imageName: string) => ({ height: 64, imageName, width: 64 });
const photo = { image: image('photo.png'), type: 'image' } as const;
const map = { bitmap: image('map.png'), offset: { x: 0, y: 0 }, type: 'paint' } as const;
const canny = { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' };
const IDENTITY = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

const layer = (patch: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => ({
  ...createControlLayer({ id: 'c1', name: 'Control Layer 1' }),
  ...patch,
});
/** A typed layer whose photo was detected into `map`. */
const detected = (patch: Partial<CanvasControlLayerContract> = {}) =>
  layer({ controlType: 'edges', detection: canny, filter: canny, original: photo, source: map, ...patch });

describe('control derivation state', () => {
  it.each([
    ['an empty legacy layer', layer(), { current: true, input: layer().source, needs: false }],
    [
      'a legacy filtered map (map-only)',
      layer({ filter: canny, source: map }),
      { current: true, input: null, needs: false },
    ],
    [
      'a typed layer using its image as-is',
      layer({ controlType: 'depth', detection: null, source: photo }),
      { current: true, input: photo, needs: false },
    ],
    [
      'a typed photo waiting for detection',
      layer({ controlType: 'edges', detection: canny, source: photo }),
      { current: false, input: photo, needs: true },
    ],
    [
      'a typed empty paint layer',
      layer({ controlType: 'edges', detection: canny }),
      { current: false, input: layer().source, needs: false },
    ],
    ['a detected layer', detected(), { current: true, input: photo, needs: false }],
    [
      'a hand-painted detected map',
      detected({ source: { ...map, bitmap: image('painted.png') } }),
      { current: true, input: photo, needs: false },
    ],
    [
      'a detected layer whose detector settings changed',
      detected({ detection: { ...canny, settings: { ...canny.settings, low_threshold: 50 } } }),
      { current: false, input: photo, needs: true },
    ],
    [
      'a typed map-only layer (filter copy) choosing a detector',
      layer({ controlType: 'edges', detection: canny, filter: canny, source: map }),
      { current: false, input: null, needs: false },
    ],
  ])('reads %s', (_label, subject, expected) => {
    expect(getControlDetectionInput(subject)).toEqual(expected.input);
    expect(isControlMapCurrent(subject)).toBe(expected.current);
    expect(needsControlDetection(subject)).toBe(expected.needs);
  });

  it('compares the producing filter with the detection structurally', () => {
    const reordered = { settings: { low_threshold: 100, high_threshold: 200 }, type: 'canny_edge_detection' };
    expect(isControlMapCurrent(detected({ filter: reordered }))).toBe(true);
  });
});

describe('controlDetectionInputKey', () => {
  const key = controlDetectionInputKey(detected());

  it('ignores weight, name and paint on the derived map', () => {
    const subject = detected();
    expect(controlDetectionInputKey({ ...subject, adapter: { ...subject.adapter, weight: 1.4 } })).toBe(key);
    expect(controlDetectionInputKey({ ...subject, name: 'Edges 1' })).toBe(key);
    expect(controlDetectionInputKey({ ...subject, source: { ...map, bitmap: image('painted.png') } })).toBe(key);
  });

  it('is independent of detector settings key order', () => {
    const reordered = { settings: { low_threshold: 100, high_threshold: 200 }, type: 'canny_edge_detection' };
    expect(controlDetectionInputKey(detected({ detection: reordered }))).toBe(key);
  });

  it.each([
    ['the input image', detected({ original: { image: image('other.png'), type: 'image' } })],
    ['the original frame', detected({ originalTransform: { ...IDENTITY, x: 4 } })],
    ['a detector setting', detected({ detection: { ...canny, settings: { ...canny.settings, low_threshold: 1 } } })],
    ['the detector', detected({ detection: { settings: {}, type: 'mlsd_detection' } })],
    ['the type', detected({ controlType: 'lineart' })],
  ])('changes with %s', (_label, subject) => {
    expect(controlDetectionInputKey(subject)).not.toBe(key);
  });

  it('follows the paint of a layer whose own pixels are the input', () => {
    const drawn = layer({ controlType: 'edges', detection: canny, source: map });
    expect(controlDetectionInputKey({ ...drawn, source: { ...map, bitmap: image('more-strokes.png') } })).not.toBe(
      controlDetectionInputKey(drawn)
    );
  });
});

describe('detachControlDerivation', () => {
  it('keeps the map as the image and makes the detection "None"', () => {
    const subject = detected({ originalTransform: { ...IDENTITY, x: 3 } });
    const detached = detachControlDerivation(subject);
    expect(detached).toEqual({ ...layer({ controlType: 'edges', filter: canny, source: map }), detection: null });
    expect(detached).not.toHaveProperty('original');
    expect(detached).not.toHaveProperty('originalTransform');
    expect(isControlMapCurrent(detached)).toBe(true);
  });

  it('returns a layer without derivation unchanged', () => {
    const legacy = layer({ filter: canny, source: map });
    expect(detachControlDerivation(legacy)).toBe(legacy);
    const none = layer({ controlType: 'depth', detection: null, source: photo });
    expect(detachControlDerivation(none)).toBe(none);
  });
});

describe('rebaseControlOriginal', () => {
  const point = { x: 7, y: -3 };
  const docPoint = (subject: CanvasControlLayerContract) =>
    applyToPoint(
      layerTransformMatrix(subject.transform),
      applyToPoint(layerTransformMatrix(subject.originalTransform ?? IDENTITY), point)
    );

  it('moves the original frame with a bake so the original lands where it was', () => {
    const before = detected({
      originalTransform: { rotation: 0.3, scaleX: 0.5, scaleY: 0.5, x: 2, y: 1 },
      transform: { rotation: 0.2, scaleX: 2, scaleY: 2, x: 30, y: 40 },
    });
    const after = rebaseControlOriginal({ ...before, transform: IDENTITY }, before.transform);
    expect(after.original).toBe(photo);
    const expected = docPoint(before);
    const actual = docPoint(after);
    expect(actual.x).toBeCloseTo(expected.x, 9);
    expect(actual.y).toBeCloseTo(expected.y, 9);
  });

  it('omits an identity frame and leaves layers without an original alone', () => {
    const identity = rebaseControlOriginal(detected({ originalTransform: { ...IDENTITY, x: 5 } }), {
      ...IDENTITY,
      x: -5,
    });
    expect(identity).not.toHaveProperty('originalTransform');
    const plain = layer({ controlType: 'edges', detection: canny, source: photo });
    expect(rebaseControlOriginal(plain, { ...IDENTITY, x: 9 })).toBe(plain);
  });

  it('detaches when the rebased frame would shear', () => {
    const subject = detected({ originalTransform: { ...IDENTITY, rotation: Math.PI / 4 } });
    const after = rebaseControlOriginal(subject, { ...IDENTITY, scaleX: 2 });
    expect(after).toEqual(detachControlDerivation(subject));
    expect(after.detection).toBeNull();
  });
});

describe('withDetectedMap', () => {
  const result = { image: image('detected.png'), rect: { height: 64, width: 64, x: -2, y: 5 } };

  it('keeps the input as the original and installs the map as a paint source', () => {
    const pending = layer({ controlType: 'edges', detection: canny, source: photo });
    const next = withDetectedMap(pending, result, canny);
    expect(next).toEqual({
      ...pending,
      filter: canny,
      original: photo,
      source: { bitmap: image('detected.png'), offset: { x: -2, y: 5 }, type: 'paint' },
    });
    expect(next && isControlMapCurrent(next)).toBe(true);
    expect(next && needsControlDetection(next)).toBe(false);
  });

  it('re-detects from the kept original, never from the previous map', () => {
    const next = withDetectedMap(detected(), result, { settings: {}, type: 'mlsd_detection' });
    expect(next?.original).toBe(photo);
    expect(next?.filter).toEqual({ settings: {}, type: 'mlsd_detection' });
  });

  it('folds the original frame into the layer transform', () => {
    const next = withDetectedMap(
      detected({ originalTransform: { ...IDENTITY, scaleX: 2, scaleY: 2 }, transform: { ...IDENTITY, x: 10 } }),
      result,
      canny
    );
    expect(next?.transform).toEqual({ rotation: 0, scaleX: 2, scaleY: 2, x: 10, y: 0 });
    expect(next).not.toHaveProperty('originalTransform');
  });

  it.each([
    ['an untyped layer', layer({ source: photo })],
    ['a layer without input pixels', layer({ controlType: 'edges', detection: canny })],
    ['a map-only layer', layer({ controlType: 'edges', detection: canny, filter: canny, source: map })],
    [
      'a frame that would shear',
      detected({
        originalTransform: { ...IDENTITY, rotation: Math.PI / 4 },
        transform: { ...IDENTITY, scaleX: 2 },
      }),
    ],
  ])('cannot land on %s', (_label, subject) => {
    expect(withDetectedMap(subject, result, canny)).toBeNull();
  });
});

describe('restoreControlOriginal', () => {
  it('returns the original to the source in its own frame and undoes a detection exactly', () => {
    const pending = layer({
      controlType: 'edges',
      detection: canny,
      source: photo,
      transform: { rotation: 0.4, scaleX: 1.5, scaleY: 1.5, x: 8, y: 9 },
    });
    const landed = withDetectedMap(
      pending,
      { image: image('detected.png'), rect: { height: 64, width: 64, x: 0, y: 0 } },
      canny
    )!;
    expect(restoreControlOriginal(landed)).toEqual({ ...pending, detection: null });
  });

  it('applies the original frame when it differs from the map', () => {
    const restored = restoreControlOriginal(
      detected({ originalTransform: { ...IDENTITY, x: 4 }, transform: { ...IDENTITY, scaleX: 2, scaleY: 2 } })
    );
    expect(restored).toMatchObject({ detection: null, source: photo, transform: { scaleX: 2, scaleY: 2, x: 8, y: 0 } });
    expect(restored).not.toHaveProperty('filter');
    expect(restored).not.toHaveProperty('original');
  });

  it('keeps the map as the image when the original frame would shear', () => {
    const subject = detected({
      originalTransform: { ...IDENTITY, rotation: Math.PI / 4 },
      transform: { ...IDENTITY, scaleX: 2 },
    });
    const restored = restoreControlOriginal(subject);
    expect(restored).toMatchObject({ detection: null, source: map, transform: subject.transform });
    expect(restored).not.toHaveProperty('original');
    expect(restored).not.toHaveProperty('filter');
  });

  it('only switches detection to "None" without an original, and leaves untyped layers alone', () => {
    const pending = layer({ controlType: 'edges', detection: canny, source: photo });
    expect(restoreControlOriginal(pending)).toEqual({ ...pending, detection: null });
    const untyped = layer({ source: photo });
    expect(restoreControlOriginal(untyped)).toBe(untyped);
  });
});
