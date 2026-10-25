/**
 * Pure rules for control layers that derive their map from a kept original; the invariants are documented on
 * `CanvasControlLayerContract`. Every function returns its input unchanged when nothing changes.
 */

import type {
  CanvasControlDetectionContract,
  CanvasControlLayerContract,
  CanvasImageRef,
  CanvasLayerSourceContract,
  CanvasPixelSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { Rect } from '@workbench/canvas-engine/types';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { composeLayerTransforms } from '@workbench/canvas-engine/transform/transformMath';

const asPixelSource = (source: CanvasLayerSourceContract): CanvasPixelSourceContract | null =>
  source.type === 'image' || source.type === 'paint' ? source : null;

const hasPixels = (source: CanvasPixelSourceContract | null): source is CanvasPixelSourceContract =>
  source !== null && (source.type === 'image' || source.bitmap !== null);

const isIdentityTransform = (transform: LayerTransform): boolean =>
  transform.x === 0 &&
  transform.y === 0 &&
  transform.rotation === 0 &&
  transform.scaleX === 1 &&
  transform.scaleY === 1;

const withoutOriginal = (layer: CanvasControlLayerContract): CanvasControlLayerContract => {
  const next = { ...layer };
  delete next.original;
  delete next.originalTransform;
  return next;
};

/**
 * The pixels detection runs on: the kept original, else the layer's own source unless a filter produced it. Null for
 * map-only layers (legacy filtered layers, filter copies) and for sources without pixels of their own.
 */
export const getControlDetectionInput = (layer: CanvasControlLayerContract): CanvasPixelSourceContract | null =>
  layer.original ?? (layer.filter ? null : asPixelSource(layer.source));

/** Whether `source` already reflects the chosen detection: no detection, or the kept original detected with it. */
export const isControlMapCurrent = (layer: CanvasControlLayerContract): boolean =>
  layer.detection === null ||
  layer.detection === undefined ||
  (layer.original !== undefined && areJsonValuesStructurallyEqual(layer.filter, layer.detection));

/** Whether a typed layer's map is stale and there are pixels to detect it from. */
export const needsControlDetection = (layer: CanvasControlLayerContract): boolean =>
  layer.controlType !== undefined &&
  layer.detection !== null &&
  layer.detection !== undefined &&
  !isControlMapCurrent(layer) &&
  hasPixels(getControlDetectionInput(layer));

const canonicalJson = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  if (value !== null && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    const entries = Object.keys(record)
      .filter((key) => record[key] !== undefined)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`);
    return `{${entries.join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
};

const inputRef = (input: CanvasPixelSourceContract | null): string => {
  if (input === null) {
    return 'none';
  }
  if (input.type === 'image') {
    return `image:${input.image.imageName}`;
  }
  return input.bitmap
    ? `paint:${input.bitmap.imageName}@${String(input.offset?.x ?? 0)},${String(input.offset?.y ?? 0)}`
    : 'paint:empty';
};

/**
 * Identifies what a detection run consumes: its input pixels and their frame, the detector with its settings, and
 * the type. Weight, name and painting on the derived map leave it unchanged, so they never stale a running job.
 */
export const controlDetectionInputKey = (layer: CanvasControlLayerContract): string =>
  canonicalJson([
    inputRef(getControlDetectionInput(layer)),
    layer.originalTransform ?? null,
    layer.detection ?? null,
    layer.controlType ?? null,
  ]);

/** Keeps the current pixels as the image itself: drops the original and makes a chosen detection "None". */
export const detachControlDerivation = (layer: CanvasControlLayerContract): CanvasControlLayerContract => {
  if (layer.original === undefined && layer.originalTransform === undefined && !layer.detection) {
    return layer;
  }
  const next = withoutOriginal(layer);
  if (next.detection !== undefined) {
    next.detection = null;
  }
  return next;
};

/**
 * Follows a bake of `bakedTransform` into the map's pixels (the layer transform then resets to identity) by moving
 * the original's frame with it. A frame the bake would shear cannot be kept, so the layer detaches instead.
 */
export const rebaseControlOriginal = (
  layer: CanvasControlLayerContract,
  bakedTransform: LayerTransform
): CanvasControlLayerContract => {
  if (layer.original === undefined) {
    return layer;
  }
  const originalTransform = layer.originalTransform
    ? composeLayerTransforms(bakedTransform, layer.originalTransform)
    : bakedTransform;
  if (!originalTransform) {
    return detachControlDerivation(layer);
  }
  if (isIdentityTransform(originalTransform)) {
    if (layer.originalTransform === undefined) {
      return layer;
    }
    const next = { ...layer };
    delete next.originalTransform;
    return next;
  }
  return { ...layer, originalTransform: { ...originalTransform } };
};

/**
 * Lands a detected map. The map's `rect` is in the detection input's local frame, so a kept `originalTransform` folds
 * into the layer transform. Null when the layer has no type (I1), no input pixels, or the fold would shear.
 */
export const withDetectedMap = (
  layer: CanvasControlLayerContract,
  map: { image: CanvasImageRef; rect: Rect },
  detection: CanvasControlDetectionContract
): CanvasControlLayerContract | null => {
  const original = getControlDetectionInput(layer);
  if (layer.controlType === undefined || !hasPixels(original)) {
    return null;
  }
  const transform = layer.originalTransform
    ? composeLayerTransforms(layer.transform, layer.originalTransform)
    : layer.transform;
  if (!transform) {
    return null;
  }
  const next: CanvasControlLayerContract = {
    ...layer,
    detection,
    filter: detection,
    original,
    source: { bitmap: { ...map.image }, offset: { x: map.rect.x, y: map.rect.y }, type: 'paint' },
    transform,
  };
  delete next.originalTransform;
  return next;
};

/**
 * Detection "None": the original becomes the source again in its own frame. When that frame would shear, the map
 * stays as the as-is image and only the derivation is dropped.
 */
export const restoreControlOriginal = (layer: CanvasControlLayerContract): CanvasControlLayerContract => {
  if (layer.original === undefined) {
    return layer.controlType === undefined || layer.detection === null ? layer : { ...layer, detection: null };
  }
  const transform = layer.originalTransform
    ? composeLayerTransforms(layer.transform, layer.originalTransform)
    : layer.transform;
  const next = withoutOriginal(
    transform ? { ...layer, detection: null, source: layer.original, transform } : { ...layer, detection: null }
  );
  delete next.filter;
  return next;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isPersistedPixelSource = (source: unknown): boolean =>
  isRecord(source) && (source.type === 'image' || (source.type === 'paint' && isRecord(source.bitmap)));

/**
 * Restores I1-I3 on an unvalidated persisted control leaf so load can repair it instead of refusing the document: a
 * null type counts as absent, an original without pixels detaches, and only bookkeeping that no longer holds is
 * dropped, never `source`. Shape validation stays with the loader.
 */
export const repairControlDerivationRecord = (record: Record<string, unknown>): Record<string, unknown> => {
  const next = { ...record };
  let changed = false;
  const drop = (key: 'controlType' | 'detection' | 'original' | 'originalTransform') => {
    if (key in next) {
      delete next[key];
      changed = true;
    }
  };
  if (next.controlType === null) {
    drop('controlType');
  }
  if (next.controlType === undefined) {
    drop('detection');
    drop('original');
    drop('originalTransform');
    return changed ? next : record;
  }
  if (next.original !== undefined && !isPersistedPixelSource(next.original)) {
    drop('original');
    if (next.detection !== undefined && next.detection !== null) {
      next.detection = null;
      changed = true;
    }
  }
  if (next.original !== undefined && !isRecord(next.detection)) {
    drop('original');
  }
  if (next.original === undefined) {
    drop('originalTransform');
  }
  return changed ? next : record;
};
