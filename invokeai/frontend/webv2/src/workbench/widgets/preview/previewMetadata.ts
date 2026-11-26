import type { TFunction } from 'i18next';

import { normalizeReferenceImageName } from '@features/generation/settings';
import { readRecordedModelRef } from '@workbench/image-actions';
import { getLegacyCanvasReferenceEntries } from '@workbench/image-actions/imageRecall';

/** Tolerant parse of untyped metadata into rows; malformed fields are skipped, the raw record stays. */

export interface PreviewMetadataEntry {
  key: string;
  label: string;
  value: string;
  /** Long free text (prompts) — render wrapped instead of truncated. */
  isMultiline?: boolean;
}

export interface PreviewMetadataGroupItem {
  key: string;
  value: string;
}

/** One row for an array kind; `count` includes items past the display cap, which `truncationNote` mentions. */
export interface PreviewMetadataGroup {
  key: string;
  label: string;
  count: number;
  items: PreviewMetadataGroupItem[];
  /** Present only when the group hides items (or its truncation note) until the user expands it. */
  collapsedItems?: PreviewMetadataGroupItem[];
  summary?: string;
  truncationNote?: string;
}

export type PreviewMetadataRow = PreviewMetadataEntry | PreviewMetadataGroup;

const ITEM_CAP = 100;
const GROUP_FULL_MAX = 5;
const GROUP_COLLAPSED_COUNT = 3;
const CANVAS_COLLAPSED_COUNT = 3;
const LAYER_TYPE_LABELS: Record<string, string> = {
  control: 'widgets.preview.layerTypes.control',
  control_layer: 'widgets.preview.layerTypes.control',
  inpaint_mask: 'widgets.preview.layerTypes.inpaint_mask',
  raster: 'widgets.preview.layerTypes.raster',
  raster_layer: 'widgets.preview.layerTypes.raster',
  regional_guidance: 'widgets.preview.layerTypes.regional_guidance',
  vector: 'widgets.preview.layerTypes.vector',
};

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

const getNumberOrScheduleDisplay = (metadata: Record<string, unknown>, key: string): string | number | null => {
  const scalar = getFiniteNumber(metadata, key);

  if (scalar !== null) {
    return scalar;
  }

  const value = metadata[key];

  return Array.isArray(value) &&
    value.length > 0 &&
    value.every((entry) => typeof entry === 'number' && Number.isFinite(entry))
    ? value.join(', ')
    : null;
};

const getModelName = (metadata: Record<string, unknown>): string | null => {
  const ref = readRecordedModelRef(metadata.model);

  return ref?.name || ref?.key || null;
};

const collapsed = (
  hidesItems: boolean,
  collapsedItems: PreviewMetadataGroupItem[]
): { collapsedItems?: PreviewMetadataGroupItem[] } => (hidesItems ? { collapsedItems } : {});

export const parsePreviewMetadata = (
  metadata: Record<string, unknown> | null,
  { kind, t }: { kind: 'image' | 'video'; t: TFunction }
): PreviewMetadataRow[] => {
  if (!isRecord(metadata)) {
    return [];
  }

  const entries: PreviewMetadataRow[] = [];
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
  const truncation = (overflow: number): { truncationNote?: string } =>
    overflow > 0 ? { truncationNote: t('widgets.preview.metadataMore', { count: overflow }) } : {};
  const layerTypeName = (type: string): string => {
    const lookup = Object.hasOwn(LAYER_TYPE_LABELS, type) ? LAYER_TYPE_LABELS[type] : undefined;
    return lookup ? t(lookup) : type.replaceAll('_', ' ');
  };
  const cfgScale = getNumberOrScheduleDisplay(metadata, 'cfg_scale');

  push('positivePrompt', t('widgets.preview.fields.prompt'), getString(metadata, 'positive_prompt'), true);
  push('negativePrompt', t('widgets.preview.fields.negativePrompt'), getString(metadata, 'negative_prompt'), true);
  push('model', t('widgets.preview.fields.model'), getModelName(metadata));
  push('seed', t('widgets.preview.fields.seed'), getFiniteNumber(metadata, 'seed'));
  push('steps', t('widgets.preview.fields.steps'), getFiniteNumber(metadata, 'steps'));
  push('cfgScale', t('widgets.preview.fields.cfgScale'), cfgScale);

  if (kind === 'image') {
    // Older distilled-guidance records store their single guidance value as `guidance` with no `cfg_scale`.
    push(
      'guidance',
      t('widgets.preview.fields.guidance'),
      cfgScale === null ? getNumberOrScheduleDisplay(metadata, 'guidance') : null
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
    const label = metadataLabel(key);
    if (label === null) {
      continue;
    }
    const items: PreviewMetadataGroupItem[] = [];
    values.slice(0, ITEM_CAP).forEach((entry, index) => {
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
        // A Generate reference the user named leads with its name.
        key === 'ref_images' && !showLegacyReferences ? (normalizeReferenceImageName(entry.name) ?? null) : null,
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
        items.push({ key: `${key}-${index}`, value: parts.filter(Boolean).join(' · ') });
      }
    });
    const overflow = Math.max(0, values.length - ITEM_CAP);
    if (items.length > 0 || overflow > 0) {
      entries.push({
        key,
        label,
        count: items.length + overflow,
        items,
        ...collapsed(items.length > GROUP_FULL_MAX || overflow > 0, items.slice(0, GROUP_COLLAPSED_COUNT)),
        ...truncation(overflow),
      });
    }
  }
  const canvas = isRecord(metadata.canvas_v3_metadata) ? metadata.canvas_v3_metadata : null;
  const document = canvas && isRecord(canvas.document) ? canvas.document : null;
  const stacks = document && isRecord(document.stacks) ? document.stacks : null;
  const legacyCanvas = isRecord(metadata.canvas_v2_metadata) ? metadata.canvas_v2_metadata : null;
  const canvasLabel = metadataLabel('canvas_layers');
  if ((stacks || legacyCanvas) && canvasLabel !== null) {
    const controlModels = canvas && isRecord(canvas.controlModels) ? canvas.controlModels : {};
    const layers: (PreviewMetadataGroupItem & { hasSettings: boolean })[] = [];
    const typeCounts = new Map<string, number>();
    let leaves = 0;
    let visited = 0;
    const visit = (nodes: unknown[], depth: number): void => {
      if (depth > 10) {
        return;
      }
      for (const raw of nodes) {
        if (visited >= 10000) {
          break;
        }
        visited += 1;
        if (!isRecord(raw)) {
          continue;
        }
        if (raw.type === 'group' && Array.isArray(raw.children)) {
          visit(raw.children, depth + 1);
          continue;
        }
        leaves += 1;
        const type = getString(raw, 'type');
        const typeName = type === null ? null : layerTypeName(type);
        if (typeName !== null) {
          typeCounts.set(typeName, (typeCounts.get(typeName) ?? 0) + 1);
        }
        if (leaves > ITEM_CAP) {
          continue;
        }
        const adapter = isRecord(raw.adapter) ? raw.adapter : isRecord(raw.controlAdapter) ? raw.controlAdapter : null;
        const model = adapter
          ? (modelName(controlModels[String(raw.id)] ?? adapter.model) ?? getString(adapter, 'model'))
          : null;
        const weight =
          adapter && typeof adapter.weight === 'number'
            ? t('widgets.preview.metadataWeight', { weight: adapter.weight })
            : null;
        const prompts = [getString(raw, 'positivePrompt'), getString(raw, 'negativePrompt')];
        const value = [getString(raw, 'name'), typeName, model, ...prompts, weight].filter(Boolean).join(' · ');
        if (value) {
          layers.push({
            key: `canvas_layers-${leaves}`,
            value,
            hasSettings: Boolean(model || weight || prompts.some(Boolean)),
          });
        }
      }
    };
    for (const [key, nodes] of Object.entries(stacks ?? legacyCanvas!)) {
      if (key !== 'referenceImages' && Array.isArray(nodes)) {
        visit(nodes, 0);
      }
    }
    const items = layers.map(({ key, value }) => ({ key, value }));
    const shownCollapsed = layers
      .filter((layer) => layer.hasSettings)
      .slice(0, CANVAS_COLLAPSED_COUNT)
      .map(({ key, value }) => ({ key, value }));
    const overflow = Math.max(0, leaves - ITEM_CAP);
    entries.push({
      key: 'canvas_layers',
      label: canvasLabel,
      count: items.length + overflow,
      items,
      ...collapsed(shownCollapsed.length < items.length || overflow > 0, shownCollapsed),
      summary: [
        t('widgets.preview.metadataLayerCount', { count: leaves }),
        ...Array.from(typeCounts, ([type, count]) => t('widgets.preview.metadataLayerTypeCount', { count, type })),
      ].join(' · '),
      ...truncation(overflow),
    });
  }

  return entries;
};
