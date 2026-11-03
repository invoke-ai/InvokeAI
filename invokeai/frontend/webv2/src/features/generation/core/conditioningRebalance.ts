/** Twelve elementwise gains, ordered from shallow to deep encoder taps. */

/**
 * The encoder layers tapped into the conditioning, shallow to deep.
 * Source of truth: `KREA2_SELECT_LAYERS` in `invokeai/backend/krea2/sampling_utils.py`.
 */
export const KREA2_TAP_LAYERS: readonly number[] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];

export const KREA2_REBALANCE_WEIGHT_COUNT: number = KREA2_TAP_LAYERS.length;

/** Backend defaults from `Krea2ConditioningRebalanceInvocation`. */
export const DEFAULT_KREA2_REBALANCE_WEIGHTS = '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0';
export const DEFAULT_KREA2_REBALANCE_MULTIPLIER = 4;

export const REBALANCE_WEIGHT_MIN = 0;
/** The nominal track ceiling does not cap legal weights; the display rescales automatically. */
export const REBALANCE_WEIGHT_TRACK_MAX = 8;
/** Weights read as "no change" here; the editor draws a rule at this level. */
export const REBALANCE_NEUTRAL_WEIGHT = 1;

/** Dragged weights quantize to one decimal, which keeps float dust out of the serialized string. */
const WEIGHT_QUANTUM = 10;
export const REBALANCE_WEIGHT_STEP = 1 / WEIGHT_QUANTUM;

/** Accept signed decimal values with optional fraction/exponent; reject JS-only hex, binary, and octal syntax. */
const DECIMAL_NUMBER_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** Parses the stored comma string, or null when the backend would reject it. */
export const parseRebalanceWeights = (value: string): number[] | null => {
  const parts = value.split(',');

  if (parts.length !== KREA2_REBALANCE_WEIGHT_COUNT) {
    return null;
  }

  const weights: number[] = [];

  for (const part of parts) {
    const trimmed = part.trim();

    if (!DECIMAL_NUMBER_RE.test(trimmed)) {
      return null;
    }

    const weight = Number(trimmed);

    // `1e999` parses to Infinity here and to `inf` in Python; the node rejects both.
    if (!Number.isFinite(weight)) {
      return null;
    }

    weights.push(weight);
  }

  return weights;
};

const formatWeight = (weight: number): string => {
  const rounded = Math.round(weight * 100) / 100;

  // Integers keep a trailing `.0` so a round trip reproduces the backend's default string.
  return Number.isInteger(rounded) ? rounded.toFixed(1) : String(rounded);
};

export const serializeRebalanceWeights = (weights: readonly number[]): string => weights.map(formatWeight).join(',');

/** Validate backend-compatible numeric syntax before forwarding raw weights. */
export const isValidKrea2RebalanceWeights = (value: string): boolean => parseRebalanceWeights(value) !== null;

const snapWeight = (weight: number): number => Math.round(weight * WEIGHT_QUANTUM) / WEIGHT_QUANTUM;

const clampWeight = (weight: number, scale: number): number => Math.min(scale, Math.max(REBALANCE_WEIGHT_MIN, weight));

/** Scale to the larger of the nominal ceiling and actual maximum. */
export const getRebalanceBarScale = (weights: readonly number[]): number =>
  weights.reduce((scale, weight) => Math.max(scale, weight), REBALANCE_WEIGHT_TRACK_MAX);

/** Puts an arbitrary weight onto the step grid and inside the track. */
export const snapRebalanceWeight = (weight: number, scale: number): number => snapWeight(clampWeight(weight, scale));

/** Pointer position to weight. `fractionFromTop` is 0 at the track's top edge, 1 at its bottom. */
export const weightFromTrackFraction = (fractionFromTop: number, scale: number): number =>
  snapRebalanceWeight((1 - fractionFromTop) * scale, scale);

/** Weight to the proportion of the track a bar fills, 0..1. */
export const barFillFraction = (weight: number, scale: number): number =>
  scale <= 0 ? 0 : clampWeight(weight, scale) / scale;

/** Applies a keyboard delta, staying on the step grid and inside the track. */
export const adjustRebalanceWeight = (weight: number, delta: number, scale: number): number =>
  snapRebalanceWeight(snapWeight(weight) + delta, scale);

const SPARKLINE_INSET = 1;

/** Inset the path so preserveAspectRatio=none does not clip its stroke. */
export const getRebalanceSparklinePath = (weights: readonly number[], width: number, height: number): string => {
  if (weights.length === 0) {
    return '';
  }

  const scale = getRebalanceBarScale(weights);
  const usableHeight = Math.max(0, height - SPARKLINE_INSET * 2);
  const stepX = weights.length === 1 ? 0 : width / (weights.length - 1);

  return weights
    .map((weight, index) => {
      const x = Math.round(index * stepX * 100) / 100;
      const y = Math.round((SPARKLINE_INSET + (1 - barFillFraction(weight, scale)) * usableHeight) * 100) / 100;

      return `${index === 0 ? 'M' : 'L'}${x} ${y}`;
    })
    .join(' ');
};

export interface RebalancePreset {
  id: string;
  label: string;
  weights: string;
  multiplier: number;
}

export const REBALANCE_PRESET_DEFAULT_ID = 'default';
export const REBALANCE_PRESET_NEUTRAL_ID = 'neutral';

/** All taps at 1.0 with unit gain — the rebalance pass as a no-op, for A/B-ing it. */
export const NEUTRAL_KREA2_REBALANCE_WEIGHTS = Array.from({ length: KREA2_REBALANCE_WEIGHT_COUNT }, () => '1.0').join(
  ','
);

/** Preset labels describe curve shapes, not measured tuning outcomes. */
export const BUILTIN_REBALANCE_PRESETS: readonly RebalancePreset[] = [
  {
    id: REBALANCE_PRESET_DEFAULT_ID,
    label: 'Default',
    multiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    weights: DEFAULT_KREA2_REBALANCE_WEIGHTS,
  },
  {
    id: REBALANCE_PRESET_NEUTRAL_ID,
    label: 'Neutral',
    multiplier: 1,
    weights: NEUTRAL_KREA2_REBALANCE_WEIGHTS,
  },
  {
    id: 'subtle',
    label: 'Subtle',
    multiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    weights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.75,3.0,1.05,2.5,1.0',
  },
  {
    id: 'strong',
    label: 'Strong',
    multiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    weights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.75,7.5,1.15,6.0,1.0',
  },
  {
    id: 'early',
    label: 'Early layers',
    multiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    weights: '4.0,3.0,2.0,1.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0',
  },
  {
    id: 'late',
    label: 'Late layers',
    multiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    weights: '1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.5,2.0,3.0,4.0',
  },
];

const BUILTIN_REBALANCE_PRESET_IDS = new Set(BUILTIN_REBALANCE_PRESETS.map((preset) => preset.id));

export const isRebalancePreset = (value: unknown): value is RebalancePreset => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const preset = value as Partial<RebalancePreset>;

  return (
    typeof preset.id === 'string' &&
    preset.id.trim() !== '' &&
    typeof preset.label === 'string' &&
    preset.label.trim() !== '' &&
    typeof preset.multiplier === 'number' &&
    Number.isFinite(preset.multiplier) &&
    typeof preset.weights === 'string' &&
    isValidKrea2RebalanceWeights(preset.weights)
  );
};

/** Discard invalid persisted curves and IDs colliding with built-ins. */
export const normalizeRebalancePresets = (value: unknown): RebalancePreset[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set<string>();
  const presets: RebalancePreset[] = [];

  for (const entry of value) {
    if (!isRebalancePreset(entry) || seen.has(entry.id) || BUILTIN_REBALANCE_PRESET_IDS.has(entry.id)) {
      continue;
    }

    seen.add(entry.id);
    presets.push({
      id: entry.id,
      label: entry.label.trim(),
      multiplier: entry.multiplier,
      weights: entry.weights,
    });
  }

  return presets;
};

/** The preset the current values match exactly, or null when they have been edited away from one. */
export const matchRebalancePreset = (
  presets: readonly RebalancePreset[],
  weights: string,
  multiplier: number
): string | null => {
  const current = parseRebalanceWeights(weights);

  if (!current) {
    return null;
  }

  const match = presets.find((preset) => {
    if (preset.multiplier !== multiplier) {
      return false;
    }

    const presetWeights = parseRebalanceWeights(preset.weights);

    return presetWeights !== null && presetWeights.every((weight, index) => weight === current[index]);
  });

  return match?.id ?? null;
};
