/**
 * Tolerant display parser for image metadata (`GET /images/i/{name}/metadata`).
 * The payload is untyped and varies by generation path (generate, canvas,
 * workflow, upscale), so every field is optional and unknown shapes are simply
 * skipped — the raw record is still available for the panel's JSON disclosure.
 */

export interface PreviewMetadataEntry {
  key: string;
  label: string;
  value: string;
  /** Long free text (prompts) — render wrapped instead of truncated. */
  isMultiline?: boolean;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const getString = (metadata: Record<string, unknown>, key: string): string | null => {
  const value = metadata[key];

  return typeof value === 'string' && value.length > 0 ? value : null;
};

const getFiniteNumber = (metadata: Record<string, unknown>, key: string): number | null => {
  const value = metadata[key];

  return typeof value === 'number' && Number.isFinite(value) ? value : null;
};

const getModelName = (metadata: Record<string, unknown>): string | null => {
  const model = metadata.model;

  if (!isRecord(model)) {
    return null;
  }

  return typeof model.name === 'string' && model.name.length > 0
    ? model.name
    : typeof model.key === 'string'
      ? model.key
      : null;
};

export const parsePreviewMetadata = (metadata: Record<string, unknown> | null): PreviewMetadataEntry[] => {
  if (!metadata) {
    return [];
  }

  const entries: PreviewMetadataEntry[] = [];
  const push = (key: string, label: string, value: string | number | null, isMultiline = false): void => {
    if (value !== null) {
      entries.push({ key, label, value: String(value), ...(isMultiline ? { isMultiline } : {}) });
    }
  };

  push('positivePrompt', 'Prompt', getString(metadata, 'positive_prompt'), true);
  push('negativePrompt', 'Negative Prompt', getString(metadata, 'negative_prompt'), true);
  push('model', 'Model', getModelName(metadata));
  push('seed', 'Seed', getFiniteNumber(metadata, 'seed'));
  push('steps', 'Steps', getFiniteNumber(metadata, 'steps'));
  push('cfgScale', 'CFG Scale', getFiniteNumber(metadata, 'cfg_scale'));
  push('scheduler', 'Scheduler', getString(metadata, 'scheduler'));
  push('clipSkip', 'CLIP Skip', getFiniteNumber(metadata, 'clip_skip'));

  const width = getFiniteNumber(metadata, 'width');
  const height = getFiniteNumber(metadata, 'height');

  if (width !== null && height !== null) {
    push('size', 'Size', `${width} x ${height}`);
  }

  return entries;
};
