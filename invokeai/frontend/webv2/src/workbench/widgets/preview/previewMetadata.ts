import type { TFunction } from 'i18next';

import { readRecordedModelRef } from '@workbench/image-actions';

/**
 * Parse variable untyped generation metadata tolerantly into display rows; skip unknown or malformed fields while the
 * complete raw record stays available for JSON display. Rows show what was recorded, whether or not recall could
 * apply it to the current panel.
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
  const ref = readRecordedModelRef(metadata.model);

  return ref?.name || ref?.key || null;
};

export const parsePreviewMetadata = (
  metadata: Record<string, unknown> | null,
  { kind, t }: { kind: 'image' | 'video'; t: TFunction }
): PreviewMetadataEntry[] => {
  if (!isRecord(metadata)) {
    return [];
  }

  const entries: PreviewMetadataEntry[] = [];
  const push = (key: string, label: string, value: string | number | null, isMultiline = false): void => {
    if (value !== null) {
      entries.push({ key, label, value: String(value), ...(isMultiline ? { isMultiline } : {}) });
    }
  };
  const cfgScale = getFiniteNumber(metadata, 'cfg_scale');

  push('positivePrompt', t('widgets.preview.fields.prompt'), getString(metadata, 'positive_prompt'), true);
  push('negativePrompt', t('widgets.preview.fields.negativePrompt'), getString(metadata, 'negative_prompt'), true);
  push('model', t('widgets.preview.fields.model'), getModelName(metadata));
  push('seed', t('widgets.preview.fields.seed'), getFiniteNumber(metadata, 'seed'));
  push('steps', t('widgets.preview.fields.steps'), getFiniteNumber(metadata, 'steps'));
  push('cfgScale', t('widgets.preview.fields.cfgScale'), cfgScale);

  if (kind === 'image') {
    // Older distilled-guidance records store their single guidance value as `guidance` with no `cfg_scale`. With both
    // recorded, which one drove the run depends on the model, which display does not consult.
    push(
      'guidance',
      t('widgets.preview.fields.guidance'),
      cfgScale === null ? getFiniteNumber(metadata, 'guidance') : null
    );
    push('scheduler', t('widgets.preview.fields.scheduler'), getString(metadata, 'scheduler'));
    push('clipSkip', t('widgets.preview.fields.clipSkip'), getFiniteNumber(metadata, 'clip_skip'));
  } else {
    push('frames', t('widgets.preview.fields.frames'), getFiniteNumber(metadata, 'num_frames'));
    push('fps', t('widgets.preview.fields.fps'), getFiniteNumber(metadata, 'fps'));
  }

  const width = getFiniteNumber(metadata, 'width');
  const height = getFiniteNumber(metadata, 'height');

  if (width !== null && height !== null) {
    push('size', t('widgets.preview.fields.size'), `${width} x ${height}`);
  }

  return entries;
};
