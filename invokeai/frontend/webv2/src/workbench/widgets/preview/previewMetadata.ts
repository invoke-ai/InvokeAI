import type { TFunction } from 'i18next';

import { readRecordedModelRef } from '@workbench/image-actions';
import { getLegacyCanvasReferenceEntries } from '@workbench/image-actions/imageRecall';

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
  const push = (key: string, label: string | null, value: string | number | null, isMultiline = false): void => {
    if (value !== null && label !== null) {
      entries.push({ key, label, value: String(value), ...(isMultiline ? { isMultiline } : {}) });
    }
  };
  // Extra labels load on demand; keep basic rows and raw metadata usable if their namespace is unavailable.
  const metadataLabel = (key: string): string | null => {
    const lookup = `widgets.preview.metadataLabels.${key}`;
    const label = t(lookup);
    return label === lookup ? null : label;
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

  const modelName = (value: unknown): string | null => {
    const ref = readRecordedModelRef(value);
    return ref?.name || ref?.key || null;
  };
  const primitiveFields = [
    'metadata_source',
    'app_version',
    'generation_mode',
    'rand_device',
    'cfg_rescale_multiplier',
    'strength',
    'seamless_x',
    'seamless_y',
    'dype_preset',
    'dype_scale',
    'dype_exponent',
    'hidiffusion',
    'hidiffusion_raunet',
    'hidiffusion_window_attn',
    'hidiffusion_t1_ratio',
    'hidiffusion_t2_ratio',
    'refiner_steps',
    'refiner_cfg_scale',
    'refiner_scheduler',
    'refiner_positive_aesthetic_score',
    'refiner_negative_aesthetic_score',
    'refiner_start',
    'refiner_denoising_start',
    'z_image_seed_variance_enabled',
    'z_image_seed_variance_strength',
    'z_image_seed_variance_randomize_percent',
    'z_image_shift',
    'qwen_image_quantization',
    'qwen_image_shift',
    'wan_guidance_scale_low_noise',
    'krea2_seed_variance_enabled',
    'krea2_seed_variance_strength',
    'krea2_seed_variance_randomize_percent',
    'krea2_rebalance_enabled',
    'krea2_rebalance_multiplier',
    'krea2_rebalance_weights',
    'krea2_reference_kv_cache',
    'anima_inpaint_adapter_weight',
    'ideogram4_sampler_preset',
    'ideogram4_steps',
    'ideogram4_guidance_scale',
    'ideogram4_mu',
    'ideogram4_caption',
    'image_size',
    'gemini_temperature',
    'gemini_thinking_level',
    'openai_quality',
    'openai_background',
    'openai_input_fidelity',
    'seedream_watermark',
    'seedream_optimize_prompt',
    'minimax_h3_duration_seconds',
    'minimax_h3_output_mode',
    'duration',
    'duration_seconds',
    'resolution',
    'aspect_ratio',
    'cfg_scale_low_noise',
    'motion_bucket_id',
    'noise_aug_strength',
    'num_inference_steps',
    'ltx2_scheduler',
    'ltx2_distilled',
    'ltx2_audio_enabled',
  ];
  const automaticFields = new Set([
    'z_image_shift',
    'qwen_image_shift',
    'wan_guidance_scale_low_noise',
    'ideogram4_steps',
    'ideogram4_guidance_scale',
    'ideogram4_mu',
    'gemini_temperature',
    'gemini_thinking_level',
    'openai_input_fidelity',
  ]);
  for (const key of primitiveFields) {
    const value = metadata[key];
    if (value === null && automaticFields.has(key)) {
      push(key, metadataLabel(key), t('common.auto'));
    }
    if (
      typeof value === 'string' ||
      (typeof value === 'number' && Number.isFinite(value)) ||
      typeof value === 'boolean'
    ) {
      push(
        key,
        metadataLabel(key),
        key === 'metadata_source' && value === 'canvas_save'
          ? t('widgets.preview.canvasSaveSource')
          : typeof value === 'boolean'
            ? t(value ? 'common.on' : 'common.off')
            : value,
        typeof value === 'string' && value.length > 100
      );
    }
  }
  const componentFields = [
    'vae',
    'refiner_model',
    't5_encoder',
    'clip_embed_model',
    'gemma2_encoder',
    'qwen3_encoder',
    'qwen3_source',
    'anima_inpaint_adapter',
    'qwen3_vl_encoder',
    'qwen3_5_encoder',
    'mistral_encoder',
    'qwen_image_component_source',
    'qwen_image_vae',
    'qwen_image_qwen_vl_encoder',
    'wan_component_source',
    'wan_transformer_low_noise',
    'wan_vae_model',
    'wan_t5_encoder_model',
    'minimax_h3_transformer_model',
    'minimax_h3_text_encoder_model',
    'ideogram4_unconditional_model',
  ];
  for (const key of componentFields) {
    push(key, metadataLabel(key), metadata[key] === null ? t('common.auto') : modelName(metadata[key]));
  }
  if (Array.isArray(metadata.ideogram4_color_palette)) {
    push(
      'ideogram4_color_palette',
      metadataLabel('ideogram4_color_palette'),
      metadata.ideogram4_color_palette.filter((term) => typeof term === 'string').join(', ')
    );
  }
  const legacyReferences = getLegacyCanvasReferenceEntries(metadata);
  const showLegacyReferences = !Array.isArray(metadata.ref_images) && Boolean(legacyReferences?.length);
  for (const key of [
    'loras',
    'ref_images',
    'control_layers',
    'controlnets',
    't2iAdapters',
    'ipAdapters',
    'regions',
  ] as const) {
    const values = key === 'ref_images' && showLegacyReferences ? legacyReferences : metadata[key];
    if (!Array.isArray(values) || values.length === 0) {
      continue;
    }
    // New records also carry legacy controlnets/t2iAdapters aliases; display each executed control just once.
    if ((key === 'controlnets' || key === 't2iAdapters') && Array.isArray(metadata.control_layers)) {
      continue;
    }
    if (key === 'ipAdapters' && (Array.isArray(metadata.ref_images) || showLegacyReferences)) {
      continue;
    }
    values.slice(0, 100).forEach((entry, index) => {
      if (!isRecord(entry)) {
        return;
      }
      const config = isRecord(entry.config) ? entry.config : isRecord(entry.ipAdapter) ? entry.ipAdapter : entry;
      const model = modelName(
        config.model ?? config.lora ?? config.control_model ?? config.t2i_adapter_model ?? config.ip_adapter_model
      );
      const image = Array.isArray(config.image) && config.image.length === 1 ? config.image[0] : config.image;
      const asset = isRecord(image) ? image : null;
      const original =
        asset && isRecord(asset.original) && isRecord(asset.original.image) ? asset.original.image : asset;
      const parts = [
        model,
        getString(config, 'type'),
        getString(config, 'imageName') ??
          getString(original ?? {}, 'image_name') ??
          (Array.isArray(config.image)
            ? config.image
                .slice(0, 100)
                .flatMap((image) => (isRecord(image) ? (getString(image, 'image_name') ?? []) : []))
                .join(', ')
            : null),
        getString(config, 'positivePrompt'),
        getString(config, 'clip_vision_model') ?? getString(config, 'clipVisionModel'),
        getString(config, 'method'),
      ];
      const weight = config.weight ?? config.control_weight;
      if (typeof weight === 'number' && Number.isFinite(weight)) {
        parts.push(t('widgets.preview.metadataWeight', { weight }));
      }
      const [begin, end] = Array.isArray(config.beginEndStepPct)
        ? config.beginEndStepPct
        : [config.begin_step_percent, config.end_step_percent];
      if (typeof begin === 'number' && Number.isFinite(begin) && typeof end === 'number' && Number.isFinite(end)) {
        parts.push(t('widgets.preview.metadataSchedule', { begin, end }));
      }
      if (parts.some(Boolean)) {
        push(`${key}-${index}`, metadataLabel(key), parts.filter(Boolean).join(' · '), true);
      }
    });
    if (values.length > 100) {
      push(`${key}-count`, metadataLabel(key), t('widgets.preview.metadataMore', { count: values.length - 100 }));
    }
  }
  const canvas = isRecord(metadata.canvas_v3_metadata) ? metadata.canvas_v3_metadata : null;
  const document = canvas && isRecord(canvas.document) ? canvas.document : null;
  const stacks = document && isRecord(document.stacks) ? document.stacks : null;
  const legacyCanvas = isRecord(metadata.canvas_v2_metadata) ? metadata.canvas_v2_metadata : null;
  if (stacks || legacyCanvas) {
    const controlModels = canvas && isRecord(canvas.controlModels) ? canvas.controlModels : {};
    let shown = 0;
    let visited = 0;
    const visit = (nodes: unknown[], depth: number): number => {
      if (depth > 10) {
        return 0;
      }
      let count = 0;
      for (const raw of nodes) {
        if (visited >= 10000) {
          break;
        }
        visited += 1;
        if (!isRecord(raw)) {
          continue;
        }
        count += 1;
        if (raw.type === 'group' && Array.isArray(raw.children)) {
          count += visit(raw.children, depth + 1);
        } else if (shown < 100) {
          shown += 1;
          const adapter = isRecord(raw.adapter)
            ? raw.adapter
            : isRecord(raw.controlAdapter)
              ? raw.controlAdapter
              : null;
          const parts = [
            getString(raw, 'name'),
            getString(raw, 'type'),
            adapter ? (modelName(controlModels[String(raw.id)] ?? adapter.model) ?? getString(adapter, 'model')) : null,
            getString(raw, 'positivePrompt'),
            getString(raw, 'negativePrompt'),
          ];
          if (adapter && typeof adapter.weight === 'number') {
            parts.push(t('widgets.preview.metadataWeight', { weight: adapter.weight }));
          }
          push(
            `canvas-${String(raw.id)}-${entries.length}`,
            metadataLabel('canvas_layers'),
            parts.filter(Boolean).join(' · '),
            true
          );
        }
      }
      return count;
    };
    const counts = Object.entries(stacks ?? legacyCanvas!).flatMap(([key, nodes]) =>
      key !== 'referenceImages' && Array.isArray(nodes) ? [visit(nodes, 0)] : []
    );
    push(
      'canvasLayerCount',
      metadataLabel('canvas_layers'),
      t('widgets.preview.metadataLayerCount', { count: counts.reduce((sum, count) => sum + count, 0) })
    );
  }

  return entries;
};
