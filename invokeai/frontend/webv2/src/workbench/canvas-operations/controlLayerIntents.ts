/**
 * Pure rules for the control layer Properties panel: each intent maps the live layer to the next one. Detection
 * itself never runs here; the coordinator schedules it when the result needs a fresh map.
 */

import type {
  CanvasControlDetectionContract,
  CanvasControlLayerContract,
  CanvasLayerSourceContract,
  ControlType,
} from '@workbench/canvas-engine/contracts';
import type { ControlLayerIntent } from '@workbench/canvas-operations/operationTypes';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { getControlDetectionInput, restoreControlOriginal } from '@workbench/canvas-engine/api';
import { CONTROL_TYPE_SPECS } from '@workbench/controlTypes';

export interface ControlLayerIntentOptions {
  /** A detector's default settings; an unknown detector has none. */
  detectorDefaults(detector: string): Record<string, unknown>;
}

const hasPixels = (source: CanvasLayerSourceContract | null): boolean =>
  source !== null && (source.type === 'image' || (source.type === 'paint' && source.bitmap !== null));

/** Whether there are pixels detection could run on (I5): map-only and empty layers have none. */
const hasDetectionPixels = (layer: CanvasControlLayerContract): boolean => hasPixels(getControlDetectionInput(layer));

const settingsMatch = (required: Record<string, unknown>, actual: Record<string, unknown>): boolean =>
  Object.entries(required).every(([key, value]) => areJsonValuesStructurallyEqual(actual[key], value));

/** The type's entry for `detector`; a type may fix some of its settings (PiDi as a sketch detector). */
const findDetectorSpec = (type: ControlType, detector: string) =>
  CONTROL_TYPE_SPECS[type].detectors.find((entry) => entry.type === detector);

const belongsToType = (type: ControlType, detection: CanvasControlDetectionContract): boolean =>
  CONTROL_TYPE_SPECS[type].detectors.some(
    (entry) => entry.type === detection.type && settingsMatch(entry.settings ?? {}, detection.settings)
  );

const detectionFor = (
  type: ControlType,
  detector: string,
  options: ControlLayerIntentOptions
): CanvasControlDetectionContract => ({
  settings: { ...options.detectorDefaults(detector), ...findDetectorSpec(type, detector)?.settings },
  type: detector,
});

/** Drops every pixel the layer derived from or was filtered from; the caller sets the new source. */
const withoutDerivedPixels = (layer: CanvasControlLayerContract): CanvasControlLayerContract => {
  const next = { ...layer };
  delete next.original;
  delete next.originalTransform;
  delete next.filter;
  return next;
};

const setType = (
  layer: CanvasControlLayerContract,
  controlType: ControlType | null,
  options: ControlLayerIntentOptions
): CanvasControlLayerContract => {
  if (controlType === null) {
    if (layer.controlType === undefined) {
      return layer;
    }
    // Restore first, so the image the user brought is what an untyped layer keeps.
    const next = { ...restoreControlOriginal(layer) };
    delete next.controlType;
    delete next.detection;
    delete next.original;
    delete next.originalTransform;
    return next;
  }
  if (layer.controlType === controlType) {
    return layer;
  }
  const typed: CanvasControlLayerContract = {
    ...layer,
    adapter: { ...layer.adapter, model: null },
    controlType,
  };
  const spec = CONTROL_TYPE_SPECS[controlType];
  let detection: CanvasControlDetectionContract | null;
  if (!hasDetectionPixels(layer)) {
    detection = null;
  } else if (layer.detection && belongsToType(controlType, layer.detection)) {
    detection = layer.detection;
  } else {
    detection = spec.defaultDetector ? detectionFor(controlType, spec.defaultDetector, options) : null;
  }
  return detection === null ? restoreControlOriginal(typed) : { ...typed, detection };
};

const setDetector = (
  layer: CanvasControlLayerContract,
  detector: string | null,
  options: ControlLayerIntentOptions
): CanvasControlLayerContract | 'no-detection-input' => {
  if (layer.controlType === undefined) {
    return layer;
  }
  if (detector === null) {
    return restoreControlOriginal(layer);
  }
  if (getControlDetectionInput(layer) === null) {
    return 'no-detection-input';
  }
  if (layer.detection?.type === detector) {
    return layer;
  }
  return { ...layer, detection: detectionFor(layer.controlType, detector, options) };
};

const setDetectorSettings = (
  layer: CanvasControlLayerContract,
  settings: Record<string, unknown>
): CanvasControlLayerContract => {
  if (layer.controlType === undefined || !layer.detection) {
    return layer;
  }
  const fixed = findDetectorSpec(layer.controlType, layer.detection.type)?.settings;
  return {
    ...layer,
    detection: { settings: { ...layer.detection.settings, ...settings, ...fixed }, type: layer.detection.type },
  };
};

const replaceImage = (
  layer: CanvasControlLayerContract,
  intent: Extract<ControlLayerIntent, { kind: 'replace-image' }>,
  options: ControlLayerIntentOptions
): CanvasControlLayerContract => {
  const next: CanvasControlLayerContract = {
    ...withoutDerivedPixels(layer),
    source: { image: { ...intent.image }, type: 'image' },
  };
  if (intent.transform) {
    next.transform = { ...intent.transform };
  }
  // None was chosen only because there was nothing to detect; a first image gets the type's detector.
  const wasEmpty = layer.original === undefined && !hasPixels(layer.source);
  const defaultDetector = layer.controlType ? CONTROL_TYPE_SPECS[layer.controlType].defaultDetector : null;
  if (layer.controlType && layer.detection === null && wasEmpty && defaultDetector) {
    next.detection = detectionFor(layer.controlType, defaultDetector, options);
  }
  return next;
};

const draw = (layer: CanvasControlLayerContract): CanvasControlLayerContract => {
  const next: CanvasControlLayerContract = { ...withoutDerivedPixels(layer), source: { bitmap: null, type: 'paint' } };
  if (next.controlType !== undefined) {
    next.detection = null;
  }
  return next;
};

/**
 * The layer after `intent`, or the same object when nothing changes. `'no-detection-input'` refuses a detector for a
 * layer whose pixels are a map without an original (a legacy filter result): replace the image first.
 */
export const applyControlLayerIntent = (
  layer: CanvasControlLayerContract,
  intent: ControlLayerIntent,
  options: ControlLayerIntentOptions
): CanvasControlLayerContract | 'no-detection-input' => {
  let next: CanvasControlLayerContract | 'no-detection-input';
  switch (intent.kind) {
    case 'set-type': {
      const typed = setType(layer, intent.controlType, options);
      next = intent.name === undefined || intent.name === typed.name ? typed : { ...typed, name: intent.name };
      break;
    }
    case 'set-detector':
      next = setDetector(layer, intent.detector, options);
      break;
    case 'set-detector-settings':
      next = setDetectorSettings(layer, intent.settings);
      break;
    case 'replace-image':
      next = replaceImage(layer, intent, options);
      break;
    case 'draw':
      next = draw(layer);
      break;
  }
  if (next === 'no-detection-input') {
    return next;
  }
  return areJsonValuesStructurallyEqual(next, layer) ? layer : next;
};
