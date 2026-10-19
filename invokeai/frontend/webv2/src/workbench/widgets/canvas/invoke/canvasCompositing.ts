/**
 * Persist canvas compositing policy per project. Generate owns numeric controls, Canvas settings owns masked-only
 * output, and prepareCanvasInvocation supplies defaulted/clamped values to compilation. Defaults mirror legacy
 * infill/coherence/mask behavior.
 */

/** The infill methods the outpaint graph can request (legacy `zInfillMethod`). */
export type CanvasInfillMethod = 'patchmatch' | 'lama' | 'cv2' | 'color' | 'tile';

/** The coherence-pass blur modes `create_gradient_mask` accepts (legacy `zParameterCanvasCoherenceMode`). */
export type CanvasCoherenceMode = 'Gaussian Blur' | 'Box Blur' | 'Staged';

/** An 8-bit RGBA color, mirroring legacy `infillColorValue` (a in 0..1). */
export interface CanvasInfillColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

/** The resolved compositing settings threaded into `compileCanvasGraph`. */
export interface CanvasCompositingSettings {
  infillMethod: CanvasInfillMethod;
  infillTileSize: number;
  infillPatchmatchDownscaleSize: number;
  infillColorValue: CanvasInfillColor;
  maskBlur: number;
  coherenceMode: CanvasCoherenceMode;
  coherenceMinDenoise: number;
  coherenceEdgeSize: number;
  /** Preserve transparency outside the generated mask instead of compositing over the source image. */
  outputOnlyMaskedRegions: boolean;
}

/** Persisted keys inside the canvas widget's `state.values`. */
export const CANVAS_COMPOSITING_KEYS = {
  coherenceEdgeSize: 'coherenceEdgeSize',
  coherenceMinDenoise: 'coherenceMinDenoise',
  coherenceMode: 'coherenceMode',
  infillColorValue: 'infillColorValue',
  infillMethod: 'infillMethod',
  infillPatchmatchDownscaleSize: 'infillPatchmatchDownscaleSize',
  infillTileSize: 'infillTileSize',
  maskBlur: 'maskBlur',
  outputOnlyMaskedRegions: 'outputOnlyMaskedRegions',
} as const;

/** Legacy-parity compositing and output defaults. */
export const DEFAULT_CANVAS_COMPOSITING: CanvasCompositingSettings = {
  coherenceEdgeSize: 16,
  coherenceMinDenoise: 0,
  coherenceMode: 'Gaussian Blur',
  infillColorValue: { a: 1, b: 0, g: 0, r: 0 },
  infillMethod: 'lama',
  infillPatchmatchDownscaleSize: 1,
  infillTileSize: 32,
  maskBlur: 16,
  outputOnlyMaskedRegions: true,
};

/** Inclusive UI/value bounds for the numeric compositing knobs. */
export const CANVAS_MASK_BLUR_MAX = 512;
export const CANVAS_COHERENCE_EDGE_SIZE_MAX = 512;

const INFILL_METHODS: ReadonlySet<CanvasInfillMethod> = new Set(['patchmatch', 'lama', 'cv2', 'color', 'tile']);
const COHERENCE_MODES: ReadonlySet<CanvasCoherenceMode> = new Set(['Gaussian Blur', 'Box Blur', 'Staged']);

export const isCanvasInfillMethod = (value: unknown): value is CanvasInfillMethod =>
  typeof value === 'string' && INFILL_METHODS.has(value as CanvasInfillMethod);

export const isCanvasCoherenceMode = (value: unknown): value is CanvasCoherenceMode =>
  typeof value === 'string' && COHERENCE_MODES.has(value as CanvasCoherenceMode);

const clampInt = (value: unknown, min: number, max: number, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;

const clampUnit = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;

const readInfillColor = (value: unknown): CanvasInfillColor => {
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    const channel = (key: string, fallback: number) => clampInt(record[key], 0, 255, fallback);
    return {
      a: clampUnit(record.a, DEFAULT_CANVAS_COMPOSITING.infillColorValue.a),
      b: channel('b', DEFAULT_CANVAS_COMPOSITING.infillColorValue.b),
      g: channel('g', DEFAULT_CANVAS_COMPOSITING.infillColorValue.g),
      r: channel('r', DEFAULT_CANVAS_COMPOSITING.infillColorValue.r),
    };
  }
  return { ...DEFAULT_CANVAS_COMPOSITING.infillColorValue };
};

/** Return complete compositing settings with legacy defaults for invalid/missing fields and valid-range clamping. */
export const readCanvasCompositingSettings = (
  values: Record<string, unknown> | undefined
): CanvasCompositingSettings => {
  const v = values ?? {};
  const d = DEFAULT_CANVAS_COMPOSITING;
  const coherenceMode = v[CANVAS_COMPOSITING_KEYS.coherenceMode];
  const infillMethod = v[CANVAS_COMPOSITING_KEYS.infillMethod];
  const outputOnlyMaskedRegions = v[CANVAS_COMPOSITING_KEYS.outputOnlyMaskedRegions];
  return {
    coherenceEdgeSize: clampInt(
      v[CANVAS_COMPOSITING_KEYS.coherenceEdgeSize],
      0,
      CANVAS_COHERENCE_EDGE_SIZE_MAX,
      d.coherenceEdgeSize
    ),
    coherenceMinDenoise: clampUnit(v[CANVAS_COMPOSITING_KEYS.coherenceMinDenoise], d.coherenceMinDenoise),
    coherenceMode: isCanvasCoherenceMode(coherenceMode) ? coherenceMode : d.coherenceMode,
    infillColorValue: readInfillColor(v[CANVAS_COMPOSITING_KEYS.infillColorValue]),
    infillMethod: isCanvasInfillMethod(infillMethod) ? infillMethod : d.infillMethod,
    infillPatchmatchDownscaleSize: clampInt(
      v[CANVAS_COMPOSITING_KEYS.infillPatchmatchDownscaleSize],
      1,
      10,
      d.infillPatchmatchDownscaleSize
    ),
    infillTileSize: clampInt(v[CANVAS_COMPOSITING_KEYS.infillTileSize], 16, 256, d.infillTileSize),
    maskBlur: clampInt(v[CANVAS_COMPOSITING_KEYS.maskBlur], 0, CANVAS_MASK_BLUR_MAX, d.maskBlur),
    outputOnlyMaskedRegions:
      typeof outputOnlyMaskedRegions === 'boolean' ? outputOnlyMaskedRegions : d.outputOnlyMaskedRegions,
  };
};
