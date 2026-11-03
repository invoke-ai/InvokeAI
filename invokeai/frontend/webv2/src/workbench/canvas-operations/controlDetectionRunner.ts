/** Runs one detector graph on a control layer's detection input and places the map in the input's local frame. */

import type {
  CanvasControlDetectionContract,
  CanvasImageRef,
  CanvasPixelSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { Rect } from '@workbench/canvas-engine/types';
import type { ControlDetectionErrorCode } from '@workbench/canvas-operations/operationTypes';
import type { BackendGraphContract } from '@workbench/graphContracts';

import { buildFilterGraph, getFilterDefinition } from '@workbench/canvas-operations/filterGraphs';
import {
  LayerFilterOutputDimensionError,
  resolveFilterOutputRect,
} from '@workbench/canvas-operations/layerFilterRunner';

/**
 * An expected detection failure with its typed code; `detail` is raw backend or engine text, never copy. Aborts
 * propagate as `AbortError` instead.
 */
export class ControlDetectionFailure extends Error {
  readonly code: ControlDetectionErrorCode;
  readonly detail: string | undefined;

  constructor(code: ControlDetectionErrorCode, detail?: string) {
    super(detail ?? code);
    this.name = 'ControlDetectionFailure';
    this.code = code;
    this.detail = detail;
  }
}

export interface ControlDetectionRunOptions {
  input: CanvasPixelSourceContract;
  detection: CanvasControlDetectionContract;
  signal: AbortSignal;
  runFilterGraph(options: {
    graph: BackendGraphContract;
    outputNodeId: string;
    signal?: AbortSignal;
  }): Promise<{ height: number; imageName: string; width: number }>;
}

export interface ControlDetectionMap {
  image: CanvasImageRef;
  /** Where the map sits in the detection input's local frame. */
  rect: Rect;
}

/** The persisted image and content rect of a detection input; null for a paint source without uploaded pixels. */
export const getControlInputImage = (input: CanvasPixelSourceContract): { imageName: string; rect: Rect } | null => {
  if (input.type === 'image') {
    return {
      imageName: input.image.imageName,
      rect: { height: input.image.height, width: input.image.width, x: 0, y: 0 },
    };
  }
  if (!input.bitmap) {
    return null;
  }
  const offset = input.offset ?? { x: 0, y: 0 };
  return {
    imageName: input.bitmap.imageName,
    rect: { height: input.bitmap.height, width: input.bitmap.width, x: offset.x, y: offset.y },
  };
};

const isAbortError = (error: unknown): boolean => error instanceof DOMException && error.name === 'AbortError';

/** Rejects as soon as `signal` aborts, so a cancelled run frees its concurrency slot without waiting on the queue. */
const raceAbort = <T>(promise: Promise<T>, signal: AbortSignal): Promise<T> =>
  new Promise<T>((resolve, reject) => {
    const onAbort = (): void => reject(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
  });

export const runControlDetection = async (options: ControlDetectionRunOptions): Promise<ControlDetectionMap> => {
  options.signal.throwIfAborted();
  const { detection } = options;
  if (!getFilterDefinition(detection.type)) {
    throw new ControlDetectionFailure('unknown-detector', detection.type);
  }
  const source = getControlInputImage(options.input);
  if (!source) {
    throw new ControlDetectionFailure('source-not-ready');
  }
  const built = buildFilterGraph(detection.type, source.imageName, detection.settings, {
    height: source.rect.height,
    preserveTransparency: true,
    width: source.rect.width,
  });
  let output: { height: number; imageName: string; width: number };
  try {
    output = await raceAbort(
      options.runFilterGraph({ graph: built.graph, outputNodeId: built.outputNodeId, signal: options.signal }),
      options.signal
    );
  } catch (error) {
    options.signal.throwIfAborted();
    if (isAbortError(error)) {
      throw error;
    }
    throw new ControlDetectionFailure('process-failed', error instanceof Error ? error.message : String(error));
  }
  options.signal.throwIfAborted();
  try {
    const rect = resolveFilterOutputRect({
      filterType: detection.type,
      output,
      settings: detection.settings,
      source: source.rect,
    });
    return { image: { height: output.height, imageName: output.imageName, width: output.width }, rect };
  } catch (error) {
    if (error instanceof LayerFilterOutputDimensionError) {
      throw new ControlDetectionFailure('output-dimensions', error.message);
    }
    throw error;
  }
};
