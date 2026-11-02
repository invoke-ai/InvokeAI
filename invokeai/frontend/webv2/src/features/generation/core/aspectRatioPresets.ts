import type { AspectRatioId, GenerateSettings } from './types';

import { ASPECT_RATIO_MAP, ASPECT_RATIO_TOLERANCE, deriveAspectRatioId } from './settings';

/**
 * A user-named aspect ratio. Only the ratio is saved, never a size: models differ in the size they work best at, so
 * picking one keeps the current pixel count. The terms are as the user wrote them and may be decimals (2.39 : 1).
 */
export interface AspectRatioPreset {
  id: string;
  label: string;
  width: number;
  height: number;
}

export type AspectRatioPresetDraft = Omit<AspectRatioPreset, 'id'>;

/** The account's saved ratios, from whichever surface edits them. */
export interface AspectRatioPresetsPort {
  /** Valid saved ratios in save order; built-ins are not included. */
  presets: readonly AspectRatioPreset[];
  /** Whether the list is full: {@link save} then adds nothing. */
  isFull: boolean;
  /** Adds a ratio under a new id and returns it; null when the list is full. */
  save(draft: AspectRatioPresetDraft): AspectRatioPreset | null;
  update(presetId: string, draft: AspectRatioPresetDraft): void;
  /** Removes a ratio and reports it with Undo, which puts it back where it was. */
  remove(presetId: string): void;
}

export interface AspectRatioTerms {
  width: number;
  height: number;
}

/** The range the built-ins span (1:8 to 8:1); generation sizes outside it are not useful. */
const MIN_ASPECT_RATIO = 1 / 8;
const MAX_ASPECT_RATIO = 8;
/** Keeps the synced preferences blob and the picker's list bounded. */
export const MAX_ASPECT_RATIO_PRESETS = 50;
export const MAX_ASPECT_RATIO_PRESET_LABEL_LENGTH = 40;
/** The largest second term {@link getAspectRatioTerms} tries before falling back to decimals. */
const MAX_INTEGER_TERM_DENOMINATOR = 32;

const isPositiveFinite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

type AspectRatioTermsError = 'terms' | 'range';

/** `terms` when a term is not a positive number, `range` when the ratio falls outside 1:8 to 8:1. */
export const getAspectRatioTermsError = (width: unknown, height: unknown): AspectRatioTermsError | null => {
  if (!isPositiveFinite(width) || !isPositiveFinite(height)) {
    return 'terms';
  }

  const ratio = width / height;

  // The tolerance keeps 1:8 and 8:1 themselves in range whatever the float error of the division.
  return ratio < MIN_ASPECT_RATIO - ASPECT_RATIO_TOLERANCE || ratio > MAX_ASPECT_RATIO + ASPECT_RATIO_TOLERANCE
    ? 'range'
    : null;
};

/** Valid saved ratios in their saved order, labels trimmed; drops malformed entries and repeated ids. */
export const normalizeAspectRatioPresets = (value: unknown): AspectRatioPreset[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set<string>();
  const presets: AspectRatioPreset[] = [];

  for (const entry of value) {
    if (presets.length >= MAX_ASPECT_RATIO_PRESETS) {
      break;
    }

    if (!entry || typeof entry !== 'object') {
      continue;
    }

    const { height, id, label, width } = entry as Partial<AspectRatioPreset>;
    const trimmed = typeof label === 'string' ? label.trim().slice(0, MAX_ASPECT_RATIO_PRESET_LABEL_LENGTH) : '';

    if (
      typeof id !== 'string' ||
      id === '' ||
      seen.has(id) ||
      trimmed === '' ||
      getAspectRatioTermsError(width, height) !== null
    ) {
      continue;
    }

    seen.add(id);
    presets.push({ height: height as number, id, label: trimmed, width: width as number });
  }

  return presets;
};

export const getAspectRatioPresetRatio = (preset: AspectRatioTerms): number => preset.width / preset.height;

type AspectRatioMatch =
  | { kind: 'preset'; preset: AspectRatioPreset }
  | { kind: 'builtin'; id: Exclude<AspectRatioId, 'Free'> }
  | null;

/**
 * What a locked ratio is called: the first saved ratio equal to it, else the built-in equal to it, else nothing. A
 * saved ratio wins over a built-in, since the user named it.
 */
export const matchAspectRatio = (ratio: number, presets: readonly AspectRatioPreset[]): AspectRatioMatch => {
  if (!isPositiveFinite(ratio)) {
    return null;
  }

  const preset = presets.find(
    (candidate) => Math.abs(getAspectRatioPresetRatio(candidate) - ratio) < ASPECT_RATIO_TOLERANCE
  );

  if (preset) {
    return { kind: 'preset', preset };
  }

  const id = deriveAspectRatioId(ratio, 1);

  return id === 'Free' ? null : { id, kind: 'builtin' };
};

const roundToHundredths = (value: number): number => Math.round(value * 100) / 100;

/**
 * Terms that describe a ratio: the smallest whole terms when the second is at most 32 (2.4 is 12 : 5), else the ratio
 * against 1 to two decimals (2.39 : 1, 1 : 1.41). Also the Add ratio form's prefill for a size, as `width / height`.
 */
export const getAspectRatioTerms = (ratio: number): AspectRatioTerms => {
  if (!isPositiveFinite(ratio)) {
    return { height: 1, width: 1 };
  }

  for (let height = 1; height <= MAX_INTEGER_TERM_DENOMINATOR; height += 1) {
    const width = Math.round(ratio * height);

    if (width > 0 && Math.abs(width / height - ratio) < ASPECT_RATIO_TOLERANCE) {
      return { height, width };
    }
  }

  return ratio >= 1
    ? { height: 1, width: roundToHundredths(ratio) }
    : { height: roundToHundredths(1 / ratio), width: 1 };
};

/** `W:H` with each term to at most two decimals in the locale's digits. */
export const formatAspectRatioTerms = (terms: AspectRatioTerms, locale?: string): string => {
  const format = new Intl.NumberFormat(locale, { maximumFractionDigits: 2, useGrouping: false });

  return `${format.format(terms.width)}:${format.format(terms.height)}`;
};

/** A ratio no saved ratio or built-in names, written as {@link getAspectRatioTerms} describes it. */
export const formatAspectRatio = (ratio: number, locale?: string): string =>
  formatAspectRatioTerms(getAspectRatioTerms(ratio), locale);

/** The Generate fields a chosen ratio is stored as: a built-in under its id and exact value, any other as locked Free. */
export const getLockedAspectRatioSettings = (
  ratio: number
): Pick<GenerateSettings, 'aspectRatioId' | 'aspectRatioIsLocked' | 'aspectRatioValue'> => {
  const id = deriveAspectRatioId(ratio, 1);

  return {
    aspectRatioId: id,
    aspectRatioIsLocked: true,
    aspectRatioValue: id === 'Free' ? ratio : ASPECT_RATIO_MAP[id].ratio,
  };
};
