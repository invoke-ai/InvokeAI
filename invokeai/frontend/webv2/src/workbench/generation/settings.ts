import type {
  AspectRatioId,
  GenerateSettings,
  GenerateWidgetValues,
  MainModelConfig,
  SupportedGenerateBase,
  VaeModelConfig,
  VaePrecision,
} from './types';

/** Schedulers accepted by the backend `denoise_latents` invocation. */
export const SCHEDULER_OPTIONS: { value: string; label: string }[] = [
  { label: 'DDIM', value: 'ddim' },
  { label: 'DDPM', value: 'ddpm' },
  { label: 'DEIS', value: 'deis' },
  { label: 'DEIS Karras', value: 'deis_k' },
  { label: 'DPM++ 2S', value: 'dpmpp_2s' },
  { label: 'DPM++ 2S Karras', value: 'dpmpp_2s_k' },
  { label: 'DPM++ 2M', value: 'dpmpp_2m' },
  { label: 'DPM++ 2M Karras', value: 'dpmpp_2m_k' },
  { label: 'DPM++ 2M SDE', value: 'dpmpp_2m_sde' },
  { label: 'DPM++ 2M SDE Karras', value: 'dpmpp_2m_sde_k' },
  { label: 'DPM++ 3M', value: 'dpmpp_3m' },
  { label: 'DPM++ 3M Karras', value: 'dpmpp_3m_k' },
  { label: 'DPM++ SDE', value: 'dpmpp_sde' },
  { label: 'DPM++ SDE Karras', value: 'dpmpp_sde_k' },
  { label: 'ER-SDE', value: 'er_sde' },
  { label: 'Euler', value: 'euler' },
  { label: 'Euler Karras', value: 'euler_k' },
  { label: 'Euler Ancestral', value: 'euler_a' },
  { label: 'Heun', value: 'heun' },
  { label: 'Heun Karras', value: 'heun_k' },
  { label: 'KDPM 2', value: 'kdpm_2' },
  { label: 'KDPM 2 Karras', value: 'kdpm_2_k' },
  { label: 'KDPM 2 Ancestral', value: 'kdpm_2_a' },
  { label: 'KDPM 2 Ancestral Karras', value: 'kdpm_2_a_k' },
  { label: 'LCM', value: 'lcm' },
  { label: 'LMS', value: 'lms' },
  { label: 'LMS Karras', value: 'lms_k' },
  { label: 'PNDM', value: 'pndm' },
  { label: 'TCD', value: 'tcd' },
  { label: 'UniPC', value: 'unipc' },
  { label: 'UniPC Karras', value: 'unipc_k' },
];

const KNOWN_SCHEDULERS = new Set(SCHEDULER_OPTIONS.map((option) => option.value));

export const isKnownScheduler = (value: string): boolean => KNOWN_SCHEDULERS.has(value);

/** Preset ratios, with the preset to switch to when dimensions are swapped. */
export const ASPECT_RATIO_MAP: Record<Exclude<AspectRatioId, 'Free'>, { ratio: number; inverseId: AspectRatioId }> = {
  '8:1': { inverseId: '1:8', ratio: 8 / 1 },
  '4:1': { inverseId: '1:4', ratio: 4 / 1 },
  '21:9': { inverseId: '9:21', ratio: 21 / 9 },
  '16:9': { inverseId: '9:16', ratio: 16 / 9 },
  '3:2': { inverseId: '2:3', ratio: 3 / 2 },
  '5:4': { inverseId: '4:5', ratio: 5 / 4 },
  '4:3': { inverseId: '3:4', ratio: 4 / 3 },
  '1:1': { inverseId: '1:1', ratio: 1 },
  '3:4': { inverseId: '4:3', ratio: 3 / 4 },
  '4:5': { inverseId: '5:4', ratio: 4 / 5 },
  '2:3': { inverseId: '3:2', ratio: 2 / 3 },
  '9:16': { inverseId: '16:9', ratio: 9 / 16 },
  '1:4': { inverseId: '4:1', ratio: 1 / 4 },
  '9:21': { inverseId: '21:9', ratio: 9 / 21 },
  '1:8': { inverseId: '8:1', ratio: 1 / 8 },
};

export const ASPECT_RATIO_OPTIONS: AspectRatioId[] = [
  'Free',
  '21:9',
  '16:9',
  '3:2',
  '4:3',
  '1:1',
  '3:4',
  '2:3',
  '9:16',
  '9:21',
];

export const isAspectRatioId = (value: unknown): value is AspectRatioId =>
  value === 'Free' || (typeof value === 'string' && value in ASPECT_RATIO_MAP);

/** SD-family models generate at multiples of 8 pixels. */
export const DIMENSION_GRID = 8;
export const MIN_DIMENSION = 64;
export const MAX_DIMENSION = 4096;

export const SEED_MAX = 4_294_967_295;

/** Max skippable CLIP layers per base (mirrors the legacy CLIP_SKIP map). */
export const CLIP_SKIP_MAX: Record<SupportedGenerateBase, number> = {
  'sd-1': 12,
  'sd-2': 24,
  sdxl: 24,
};

export const getOptimalDimension = (base: string): number => (base === 'sd-1' || base === 'sd-2' ? 512 : 1024);

export const clampDimension = (value: number): number =>
  Math.min(MAX_DIMENSION, Math.max(MIN_DIMENSION, Math.round(value / DIMENSION_GRID) * DIMENSION_GRID));

/** Fit a width/height of the given ratio into approximately the given pixel area, snapped to the grid. */
export const calculateNewSize = (ratio: number, area: number): { width: number; height: number } => {
  const exactWidth = Math.sqrt(area * ratio);

  return { height: clampDimension(exactWidth / ratio), width: clampDimension(exactWidth) };
};

/** Best-matching preset for existing dimensions, falling back to Free. */
export const deriveAspectRatioId = (width: number, height: number): AspectRatioId => {
  if (height <= 0) {
    return 'Free';
  }

  const ratio = width / height;

  for (const [id, entry] of Object.entries(ASPECT_RATIO_MAP)) {
    if (Math.abs(entry.ratio - ratio) < 0.0001) {
      return id as AspectRatioId;
    }
  }

  return 'Free';
};

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object';

const hasFiniteNumber = (record: Record<string, unknown>, key: string): boolean =>
  typeof record[key] === 'number' && Number.isFinite(record[key]);

export const isMainModelConfig = (value: unknown): value is MainModelConfig =>
  isRecord(value) && typeof value.key === 'string' && typeof value.name === 'string' && value.type === 'main';

export const isVaeModelConfig = (value: unknown): value is VaeModelConfig =>
  isRecord(value) && typeof value.key === 'string' && typeof value.name === 'string' && value.type === 'vae';

const isVaePrecision = (value: unknown): value is VaePrecision => value === 'fp16' || value === 'fp32';

/**
 * Validate stored widget values and fill in fields that older persisted
 * projects predate. The core fields must be present and well-typed; newer
 * fields fall back to their defaults so old projects keep their prompts and
 * dimensions instead of being reset wholesale.
 */
export const normalizeGenerateSettings = (values: unknown): GenerateSettings | null => {
  if (!isRecord(values)) {
    return null;
  }

  const isCoreShapeValid =
    typeof values.modelKey === 'string' &&
    typeof values.positivePrompt === 'string' &&
    typeof values.negativePrompt === 'string' &&
    typeof values.scheduler === 'string' &&
    typeof values.shouldRandomizeSeed === 'boolean' &&
    ['batchCount', 'width', 'height', 'steps', 'cfgScale', 'cfgRescaleMultiplier', 'seed'].every((key) =>
      hasFiniteNumber(values, key)
    );

  if (!isCoreShapeValid) {
    return null;
  }

  const width = values.width as number;
  const height = values.height as number;

  return {
    aspectRatioId: isAspectRatioId(values.aspectRatioId) ? values.aspectRatioId : deriveAspectRatioId(width, height),
    aspectRatioIsLocked: typeof values.aspectRatioIsLocked === 'boolean' ? values.aspectRatioIsLocked : false,
    aspectRatioValue:
      hasFiniteNumber(values, 'aspectRatioValue') && (values.aspectRatioValue as number) > 0
        ? (values.aspectRatioValue as number)
        : height > 0
          ? width / height
          : 1,
    batchCount: values.batchCount as number,
    cfgRescaleMultiplier: values.cfgRescaleMultiplier as number,
    cfgScale: values.cfgScale as number,
    clipSkip: hasFiniteNumber(values, 'clipSkip') ? (values.clipSkip as number) : 0,
    height,
    modelKey: values.modelKey as string,
    negativePrompt: values.negativePrompt as string,
    positivePrompt: values.positivePrompt as string,
    scheduler: values.scheduler as string,
    seamlessXAxis: typeof values.seamlessXAxis === 'boolean' ? values.seamlessXAxis : false,
    seamlessYAxis: typeof values.seamlessYAxis === 'boolean' ? values.seamlessYAxis : false,
    seed: values.seed as number,
    shouldRandomizeSeed: values.shouldRandomizeSeed as boolean,
    steps: values.steps as number,
    vae: isVaeModelConfig(values.vae) ? values.vae : null,
    vaePrecision: isVaePrecision(values.vaePrecision) ? values.vaePrecision : 'fp32',
    width,
  };
};

export const normalizeGenerateWidgetValues = (values: unknown): GenerateWidgetValues | null => {
  const settings = normalizeGenerateSettings(values);

  if (!settings || !isRecord(values) || !isMainModelConfig(values.model)) {
    return null;
  }

  return { ...settings, model: values.model };
};

export const isGenerateSettings = (values: unknown): values is GenerateSettings => {
  const normalized = normalizeGenerateSettings(values);

  if (!normalized || !isRecord(values)) {
    return false;
  }

  // Strict only over the keys normalize would have to invent.
  return (
    isAspectRatioId(values.aspectRatioId) &&
    typeof values.aspectRatioIsLocked === 'boolean' &&
    hasFiniteNumber(values, 'aspectRatioValue') &&
    hasFiniteNumber(values, 'clipSkip') &&
    typeof values.seamlessXAxis === 'boolean' &&
    typeof values.seamlessYAxis === 'boolean' &&
    isVaePrecision(values.vaePrecision) &&
    (values.vae === null || isVaeModelConfig(values.vae))
  );
};

export const isGenerateWidgetValues = (values: unknown): values is GenerateWidgetValues =>
  isGenerateSettings(values) && isMainModelConfig((values as unknown as Record<string, unknown>).model);
