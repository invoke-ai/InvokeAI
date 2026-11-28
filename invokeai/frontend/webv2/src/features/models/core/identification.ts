import type { ModelConfig, ModelIdentification } from '@features/models/core/types';

import { getModelBaseLabel } from '@features/models/core/baseIdentity';
import { getModelFormatLabel, getModelTypeLabel, getModelVariantLabel } from '@features/models/core/taxonomy';

type IdentityFields = Pick<ModelConfig, 'base' | 'format' | 'path' | 'type'>;

/**
 * Keys per bulk identification request. The backend accepts 100, but a request is one blocking call and pickled
 * checkpoints are scanned and unpickled, so small batches keep progress visible and a stop prompt.
 */
export const IDENTIFICATION_BATCH_SIZE = 10;

/**
 * Whether the preview is cheap and useful enough to run without being asked. Probing reads only the header of
 * safetensors and GGUF files but scans and unpickles `.ckpt`/`.pt`/`.bin` ones (a legacy non-zip pickle is read
 * whole), and a folder model reads its configs; those wait for the user. An Unknown record is always worth a
 * look: that is what the preview is for.
 */
export const shouldAutoIdentify = (model: IdentityFields): boolean => {
  if (model.type === 'unknown') {
    return true;
  }

  const path = model.path.toLowerCase();

  return model.format === 'gguf' || path.endsWith('.safetensors') || path.endsWith('.gguf');
};

export type IdentityField = 'base' | 'format' | 'type' | 'variant';

export interface IdentityChange {
  field: IdentityField;
  from: string | null;
  to: string | null;
}

const IDENTITY_FIELDS: readonly IdentityField[] = ['type', 'base', 'format', 'variant'];

const labelOf = (field: IdentityField, model: Pick<ModelConfig, IdentityField>): string | null => {
  switch (field) {
    case 'base':
      return getModelBaseLabel(model.base);
    case 'format':
      return getModelFormatLabel(model.format);
    case 'type':
      return getModelTypeLabel(model.type);
    case 'variant':
      return model.variant ? getModelVariantLabel(model.variant) : null;
  }
};

/** The identity fields that differ between the stored record and the proposal, as display labels. */
export const getIdentityChanges = (
  current: Pick<ModelConfig, IdentityField>,
  proposed: Pick<ModelConfig, IdentityField>
): IdentityChange[] =>
  IDENTITY_FIELDS.flatMap((field) => {
    const from = labelOf(field, current);
    const to = labelOf(field, proposed);

    return from === to ? [] : [{ field, from, to }];
  });

/** Short name of what the proposal is, such as "SDXL Main Model". */
export const getProposedModelLabel = (proposed: Pick<ModelConfig, 'base' | 'type'>): string =>
  `${getModelBaseLabel(proposed.base)} ${getModelTypeLabel(proposed.type)}`;

/** Splits keys into request-sized batches, in order. */
export const chunkKeys = (keys: readonly string[], size = IDENTIFICATION_BATCH_SIZE): string[][] => {
  const chunks: string[][] = [];

  for (let index = 0; index < keys.length; index += size) {
    chunks.push(keys.slice(index, index + size));
  }

  return chunks;
};

export const hasProposal = (
  identification: ModelIdentification
): identification is ModelIdentification & { proposed: ModelConfig } => identification.proposed !== null;
