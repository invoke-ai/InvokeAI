import type { GenerateSettings } from './types';

/**
 * Best-effort provenance for generate-compiled graphs: which Generate-panel
 * setting produced a node input. Keyed by the builders' fixed node ids
 * (graph.ts:313-328) — deliberately not exhaustive; unmapped fields simply
 * show no "Set by" entry in the preview inspector.
 */

export interface GenerateProvenanceEntry {
  labelKey: string;
  settingKey: keyof GenerateSettings;
}

const BY_NODE_AND_FIELD: Record<string, GenerateProvenanceEntry> = {
  'clip_skip.skipped_layers': { labelKey: 'graphPreview.provenance.clipSkip', settingKey: 'clipSkip' },
  'denoise_latents.cfg_scale': { labelKey: 'graphPreview.provenance.cfgScale', settingKey: 'cfgScale' },
  'denoise_latents.scheduler': { labelKey: 'graphPreview.provenance.scheduler', settingKey: 'scheduler' },
  'denoise_latents.steps': { labelKey: 'graphPreview.provenance.steps', settingKey: 'steps' },
  'model_loader.model': { labelKey: 'graphPreview.provenance.model', settingKey: 'modelKey' },
  'negative_prompt.value': { labelKey: 'graphPreview.provenance.negativePrompt', settingKey: 'negativePrompt' },
  'noise.height': { labelKey: 'graphPreview.provenance.size', settingKey: 'height' },
  'noise.width': { labelKey: 'graphPreview.provenance.size', settingKey: 'width' },
  'positive_prompt.value': { labelKey: 'graphPreview.provenance.positivePrompt', settingKey: 'positivePrompt' },
  'seed.value': { labelKey: 'graphPreview.provenance.seed', settingKey: 'seed' },
};

const LORA_FIELDS = new Set(['lora', 'weight']);

export const getGenerateNodeProvenance = (nodeId: string, fieldName: string): GenerateProvenanceEntry | null => {
  if (nodeId.startsWith('lora_selector') && LORA_FIELDS.has(fieldName)) {
    return { labelKey: 'graphPreview.provenance.loras', settingKey: 'loras' };
  }

  return BY_NODE_AND_FIELD[`${nodeId}.${fieldName}`] ?? null;
};
