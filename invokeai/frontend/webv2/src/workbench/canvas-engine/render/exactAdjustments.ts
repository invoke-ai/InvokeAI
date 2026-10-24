/**
 * Exact adjustment evaluation for pixel-writing consumers (generation, exports, bakes, PSD, filter input,
 * extraction). Per-pixel runs apply on the calling thread; spatial runs go to the shared spatial service at `now`
 * priority. A stack is never evaluated with its spatial entries left out: when they cannot run, the consumer refuses
 * with an {@link AdjustmentRefusal} naming the owner and the cause.
 */

import type { CanvasAdjustmentsContract, CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type {
  SpatialRunPriority,
  SpatialRunRequest,
  SpatialRunResult,
} from '@workbench/canvas-engine/render/spatial/spatialAdjustmentService';
import type { Rect } from '@workbench/canvas-engine/types';

import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import {
  type AdjustablePixels,
  type AdjustmentOrigin,
  applyAdjustments,
  toSpatialPlan,
} from '@workbench/canvas-engine/render/adjustments';

/** Why an owner's spatial adjustments could not run, as the spatial service reports it. */
export type AdjustmentRefusalReason = 'too-large' | 'over-budget' | 'error';

/** A pixel-writing operation refused because one owner's spatial adjustments could not be evaluated exactly. */
export interface AdjustmentRefusal {
  readonly status: 'adjustment-refused';
  /** The layer or group whose stack could not run. */
  readonly ownerId: string;
  readonly ownerName: string;
  readonly reason: AdjustmentRefusalReason;
}

/** Evaluates spatial steps off the main thread (`SpatialAdjustmentService.run`). */
export type SpatialRunner = (request: SpatialRunRequest) => Promise<SpatialRunResult>;

export interface ExactAdjustmentOptions {
  /** Without a runner a spatial stack refuses `error` rather than leave its spatial entries out. */
  readonly run?: SpatialRunner;
  readonly signal?: AbortSignal;
}

export type ExactAdjustmentResult =
  | { readonly status: 'ok'; readonly data: Uint8ClampedArray }
  | { readonly status: 'aborted' | AdjustmentRefusalReason };

/**
 * Thrown by renderers that report through exceptions (the raster composite) when an owner's stack cannot run; the
 * operation boundary turns it into an {@link AdjustmentRefusal} with the owner's name.
 */
export class AdjustmentRefusedError extends Error {
  constructor(
    readonly ownerId: string,
    readonly reason: AdjustmentRefusalReason
  ) {
    super(`The adjustments of ${ownerId} could not be applied (${reason}).`);
    this.name = 'AdjustmentRefusedError';
  }
}

/** The refusal for `ownerId`, named from `document` (the id when the owner is gone). */
export const adjustmentRefusal = (
  document: CanvasDocumentContractV3 | null,
  ownerId: string,
  reason: AdjustmentRefusalReason
): AdjustmentRefusal => ({
  ownerId,
  ownerName: (document && getDocumentIndex(document).byId.get(ownerId)?.node.name) || ownerId,
  reason,
  status: 'adjustment-refused',
});

/** Type guard for results that mix string statuses with an {@link AdjustmentRefusal}. */
export const isAdjustmentRefusal = (value: unknown): value is AdjustmentRefusal =>
  typeof value === 'object' && value !== null && (value as { status?: unknown }).status === 'adjustment-refused';

export interface AdjustmentStackOptions {
  /** Without a runner a stack with spatial entries resolves `error`. */
  readonly run?: SpatialRunner;
  readonly signal?: AbortSignal;
  readonly priority: SpatialRunPriority;
  /** The service slot; a later request for the same slot supersedes a queued one. */
  readonly slotKey: string;
}

/**
 * Evaluates `adjustments` in list order on `pixels`, whose (0,0) sits at `origin` in the stack's reference space:
 * per-pixel runs on this thread, each run of spatial entries as one service job. Spatial runs need `pixels` to be the
 * owner's whole source (their domain is its alpha bounds) and `pixels.data` to own its entire buffer, which is
 * transferred to the worker: afterwards only the returned pixels may be used. A stack without spatial entries is
 * applied in place. Reports the service's outcome unchanged, `superseded` included.
 */
export const evaluateAdjustmentStack = async (
  pixels: AdjustablePixels,
  adjustments: CanvasAdjustmentsContract | undefined,
  origin: AdjustmentOrigin,
  options: AdjustmentStackOptions
): Promise<SpatialRunResult> => {
  const { height, width } = pixels;
  let data = pixels.data;
  for (const run of toSpatialPlan(adjustments)) {
    if (run.kind === 'per-pixel') {
      applyAdjustments({ data, height, width }, run.adjustments, origin);
      continue;
    }
    if (!options.run) {
      return { status: 'error' };
    }
    if (options.signal?.aborted) {
      return { status: 'aborted' };
    }
    const result = await options.run({
      height,
      pixels: data,
      priority: options.priority,
      signal: options.signal,
      slotKey: options.slotKey,
      steps: run.steps,
      width,
    });
    if (result.status !== 'ok') {
      return result;
    }
    data = result.pixels;
  }
  return { pixels: data, status: 'ok' };
};

// Every exact job gets its own slot, so concurrent consumers never supersede each other in the shared service.
let nextJob = 0;

/**
 * {@link evaluateAdjustmentStack} for pixel-writing consumers: spatial runs go at `now` priority, each job in its own
 * service slot.
 */
export const adjustExactly = async (
  pixels: AdjustablePixels,
  adjustments: CanvasAdjustmentsContract | undefined,
  origin: AdjustmentOrigin,
  options: ExactAdjustmentOptions
): Promise<ExactAdjustmentResult> => {
  const result = await evaluateAdjustmentStack(pixels, adjustments, origin, {
    ...options,
    priority: 'now',
    slotKey: `exact:${(nextJob += 1)}`,
  });
  if (result.status === 'ok') {
    return { data: result.pixels, status: 'ok' };
  }
  // Slots are unique, so nothing supersedes an exact job; treat it as a failure should it ever happen.
  return { status: result.status === 'superseded' ? 'error' : result.status };
};

/** `ImageData` is absent on the Node raster stub; a structural stand-in carries the same fields. */
export const imageDataOf = (data: Uint8ClampedArray, width: number, height: number): ImageData =>
  typeof ImageData === 'undefined'
    ? ({ colorSpace: 'srgb', data, height, width } as ImageData)
    : new ImageData(data as Uint8ClampedArray<ArrayBuffer>, width, height);

/**
 * A new `rect`-sized surface holding `source`'s pixels at (0,0) with `adjustments` evaluated exactly; `rect`'s origin
 * places it in the stack's reference space. Allocates the surface and one ImageData buffer (8 B/px), which the
 * caller reserves; a spatial job reserves its in-flight buffer separately.
 */
export const adjustSurfaceExactly = async (
  createSurface: (width: number, height: number) => RasterSurface,
  source: RasterSurface,
  rect: Rect,
  adjustments: CanvasAdjustmentsContract | undefined,
  options: ExactAdjustmentOptions
): Promise<
  { readonly status: 'ok'; readonly surface: RasterSurface } | Exclude<ExactAdjustmentResult, { status: 'ok' }>
> => {
  const { height, width } = rect;
  const surface = createSurface(width, height);
  const ctx = surface.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source.canvas, 0, 0);
  const image = ctx.getImageData(0, 0, width, height);
  const result = await adjustExactly(image, adjustments, rect, options);
  if (result.status !== 'ok') {
    return result;
  }
  ctx.putImageData(result.data === image.data ? image : imageDataOf(result.data, width, height), 0, 0);
  return { status: 'ok', surface };
};
