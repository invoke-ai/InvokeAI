import type { GalleryImage } from '@features/gallery';
import type { GenerateModelConfig, ComponentModelConfig } from '@features/generation/contracts';
import type { CanvasDocumentContractV3, CanvasLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryDurability, galleryImages } from '@features/gallery';
import { galleryImageUrls } from '@features/gallery/utility';
import {
  getRegionalGuidanceSupport,
  isControlKindSupportedForBase,
  getControlValidationReason,
} from '@features/generation/graph';
import { createItemId } from '@platform/browser/randomUuid';
import { assertAccountScopeCurrent, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { prepareLegacyCanvasMetadata, getCanvasEngine } from '@workbench/canvas-operations/api';
import { createEmptyCanvasState, isRecord, loadCanvasState } from '@workbench/canvasMigration';
import { deriveControlKind, isStorableControlWeight, isStorableControlStepRange } from '@workbench/controlAdapters';
import {
  readCanvasCompositingSettings,
  CANVAS_COMPOSITING_KEYS,
} from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { readCanvasScaling, CANVAS_SCALING_KEYS } from '@workbench/widgets/canvas/invoke/canvasScaling';

import { resolveRecordedModel, toRecordedModelRef } from './recordedModel';

// Matches MAX_IMAGE_BATCH_SIZE in invokeai/app/api/routers/images.py.
const IMAGE_LOOKUP_BATCH_SIZE = 1000;

export interface CanvasRecallTarget {
  commands: Pick<WorkbenchCommands, 'canvas' | 'widgets'>;
  /** Null when the originating project is closed or no longer active. */
  getProject(): Project | null;
  projectId: string;
}

export interface CanvasRecallPlan {
  document: CanvasDocumentContractV3 | null;
  /** Only acknowledged intermediate uploads produced by this preparation, never borrowed assets. */
  createdImageNames?: readonly string[];
  /** Partial recall merges instead of replacing authored work. */
  partial: boolean;
  controlsOnly?: boolean;
  /** Standalone executed maps have frame-relative placement, never authored Canvas coordinates. */
  fitControlsToFrame?: boolean;
  problems: string[];
  settings: Record<string, unknown>;
}

export const hasCanvasMetadata = (metadata: unknown): boolean =>
  isRecord(metadata) &&
  ('canvas_v3_metadata' in metadata ||
    'canvas_v2_metadata' in metadata ||
    'control_layers' in metadata ||
    'controlnets' in metadata ||
    't2iAdapters' in metadata);

export const readCanvasStrength = (metadata: unknown): number | null => {
  if (!isRecord(metadata)) {
    return null;
  }
  const record = isRecord(metadata.canvas_v3_metadata) ? metadata.canvas_v3_metadata : null;
  const settings = record && isRecord(record.settings) ? record.settings : null;
  const value = metadata.denoise_strength ?? metadata.strength ?? settings?.strength;
  return typeof value === 'number' && Number.isFinite(value) && value >= 0.01 && value <= 1 ? value : null;
};

/** Never default missing settings: a sparse old record must not reset unrelated canvas preferences. */
const readCanvasSettings = (metadata: Record<string, unknown>): Record<string, unknown> => {
  const patch: Record<string, unknown> = {};
  const strength = readCanvasStrength(metadata);
  if (strength !== null) {
    patch.denoisingStrength = strength;
  }
  const record = isRecord(metadata.canvas_v3_metadata) ? metadata.canvas_v3_metadata : null;
  const settings = record && isRecord(record.settings) ? record.settings : null;
  if (settings && isRecord(settings.compositing)) {
    const raw = settings.compositing;
    const normalized = readCanvasCompositingSettings(raw);
    for (const key of Object.values(CANVAS_COMPOSITING_KEYS)) {
      const value = raw[key],
        canonical = normalized[key];
      if (value === undefined) {
        continue;
      }
      if (key === 'infillColorValue') {
        if (
          isRecord(value) &&
          typeof canonical === 'object' &&
          ['r', 'g', 'b', 'a'].every((channel) => value[channel] === canonical[channel as keyof typeof canonical])
        ) {
          patch[key] = canonical;
        }
      } else if (value === canonical) {
        patch[key] = canonical;
      }
    }
  }
  if (settings && isRecord(settings.scaling)) {
    const raw = settings.scaling;
    const normalized = readCanvasScaling({ scaleMethod: raw.method, scaledWidth: raw.width, scaledHeight: raw.height });
    for (const side of ['method', 'width', 'height'] as const) {
      if (raw[side] !== undefined && raw[side] === normalized[side]) {
        patch[CANVAS_SCALING_KEYS[side]] = normalized[side];
      }
    }
  }
  return patch;
};

const resolveModel = (value: unknown, models: readonly ComponentModelConfig[]): ComponentModelConfig | null => {
  const ref = typeof value === 'string' ? toRecordedModelRef({ key: value }) : toRecordedModelRef(value);
  const model = ref ? resolveRecordedModel(ref, models, { ambiguous: 'refuse', hashMiss: 'refuse' }) : null;
  if (
    !model ||
    model === 'ambiguous' ||
    (ref?.base && ref.base !== model.base) ||
    (ref?.type && ref.type !== model.type)
  ) {
    return null;
  }
  return model;
};

const sourceNames = (layer: CanvasLayerContract): string[] => {
  const names: string[] = [];
  if ('source' in layer) {
    if (layer.source.type === 'image') {
      names.push(layer.source.image.imageName);
    }
    if (layer.source.type === 'paint' && layer.source.bitmap) {
      names.push(layer.source.bitmap.imageName);
    }
  } else if (layer.mask.bitmap) {
    names.push(layer.mask.bitmap.imageName);
  }
  if (layer.type === 'control' && layer.original) {
    if (layer.original.type === 'image') {
      names.push(layer.original.image.imageName);
    }
    if (layer.original.type === 'paint' && layer.original.bitmap) {
      names.push(layer.original.bitmap.imageName);
    }
  }
  if (layer.type === 'regional_guidance') {
    for (const ref of layer.referenceImages) {
      if (ref.config.image) {
        names.push(ref.config.image.imageName);
      }
    }
  }
  return names;
};

/** Bounded bulk asset lookups, strict document validation, portable model resolution, no project mutation. */
export const prepareCanvasRecall = async ({
  metadata,
  model,
  models,
  owner,
  isCurrent,
  resolvedImages = new Map<string, GalleryImage>(),
}: {
  isCurrent?: () => boolean;
  /** Internal recursion reuses asset reads from standalone control projection. */
  resolvedImages?: Map<string, GalleryImage>;
  metadata: unknown;
  model: GenerateModelConfig;
  models: readonly ComponentModelConfig[];
  owner: AccountScope;
}): Promise<CanvasRecallPlan> => {
  const plan: CanvasRecallPlan = { document: null, partial: false, problems: [], settings: {} };
  if (!isRecord(metadata)) {
    return plan;
  }
  plan.settings = readCanvasSettings(metadata);
  if (('strength' in metadata || 'denoise_strength' in metadata) && readCanvasStrength(metadata) === null) {
    plan.problems.push('invalid-strength');
  }
  if (!hasCanvasMetadata(metadata)) {
    return plan;
  }
  // Bound JSON as well as the canvas loader's 10,000 nodes / ten levels. Never traverse arbitrary metadata recursively.
  if (JSON.stringify(metadata).length > 32 * 1024 * 1024) {
    plan.problems.push('record-too-large');
    return plan;
  }
  const lookupImages = async (names: string[]): Promise<boolean> => {
    if (isCurrent && !isCurrent()) {
      plan.settings = {};
      plan.problems.push('canvas-changed');
      return false;
    }
    const missing = [...new Set(names)].filter((name) => !resolvedImages.has(name));
    assertAccountScopeCurrent(owner);
    for (let offset = 0; offset < missing.length; offset += IMAGE_LOOKUP_BATCH_SIZE) {
      try {
        const images = await galleryImages.resolveMany(
          missing.slice(offset, offset + IMAGE_LOOKUP_BATCH_SIZE),
          owner.signal
        );
        assertAccountScopeCurrent(owner);
        if (isCurrent && !isCurrent()) {
          plan.settings = {};
          plan.problems.push('canvas-changed');
          return false;
        }
        for (const image of images) {
          resolvedImages.set(image.imageName, image);
        }
      } catch (error) {
        assertAccountScopeCurrent(owner);
        if (isCurrent && !isCurrent()) {
          plan.settings = {};
          plan.problems.push('canvas-changed');
          return false;
        }
        const networkFailure =
          error instanceof TypeError &&
          [
            'Failed to fetch',
            'NetworkError when attempting to fetch resource.',
            'Load failed',
            'fetch failed',
          ].includes(error.message);
        if (!(error instanceof ApiError) && !networkFailure) {
          throw error;
        }
        plan.settings = {};
        plan.partial = true;
        plan.problems.push('image-lookup-failed');
        return false;
      }
    }
    return true;
  };
  const record = metadata.canvas_v3_metadata;
  if (!isRecord(record) || record.version !== 1) {
    if ('canvas_v3_metadata' in metadata) {
      plan.problems.push('invalid-record');
      return plan;
    }
    let legacyMetadata = metadata;
    const standalone = !('canvas_v2_metadata' in metadata) && !('control_layers' in metadata);
    if (!('canvas_v2_metadata' in metadata)) {
      let controls: unknown[];
      if (standalone) {
        controls = [];
        for (const key of ['controlnets', 't2iAdapters'] as const) {
          if (!(key in metadata)) {
            continue;
          }
          const entries = metadata[key];
          if (!Array.isArray(entries)) {
            plan.problems.push('invalid-control');
            continue;
          }
          if (controls.length + entries.length > 10000) {
            plan.problems.push('record-too-large');
          }
          for (const entry of entries.slice(0, 10000 - controls.length)) {
            controls.push(
              isRecord(entry)
                ? {
                    ...entry,
                    model: entry[key === 'controlnets' ? 'control_model' : 't2i_adapter_model'],
                    weight: key === 'controlnets' ? entry.control_weight : entry.weight,
                  }
                : entry
            );
          }
        }
        // Old aliases describe executed adapters; an empty or damaged list never means Clear Canvas.
        if (controls.length === 0) {
          plan.problems.push('invalid-control');
          return plan;
        }
      } else {
        if (!Array.isArray(metadata.control_layers)) {
          plan.problems.push('invalid-control');
          return plan;
        }
        if (metadata.control_layers.length > 10000) {
          plan.problems.push('record-too-large');
        }
        controls = metadata.control_layers.slice(0, 10000);
      }
      const names = controls.flatMap((control) => {
        const asset = isRecord(control) ? (control.processed_image ?? control.image) : null;
        return isRecord(asset) && typeof asset.image_name === 'string' && asset.image_name ? [asset.image_name] : [];
      });
      if (!(await lookupImages(names))) {
        return plan;
      }
      controls = controls.map((control) => {
        if (!isRecord(control)) {
          return control;
        }
        if (
          standalone &&
          control.control_mode !== undefined &&
          !['balanced', 'more_prompt', 'more_control', 'unbalanced'].includes(String(control.control_mode))
        ) {
          plan.problems.push('invalid-control');
          return null;
        }
        if (standalone && Array.isArray(control.weight)) {
          plan.problems.push('unsupported-control');
          return null;
        }
        if (
          standalone &&
          control.resize_mode !== undefined &&
          control.resize_mode !== 'just_resize' &&
          control.resize_mode !== 'just_resize_simple'
        ) {
          plan.problems.push('unsupported-control');
          return null;
        }
        const asset = control.processed_image ?? control.image;
        const resolved =
          isRecord(asset) && typeof asset.image_name === 'string' ? resolvedImages.get(asset.image_name) : null;
        if (!resolved) {
          plan.problems.push('invalid-control-image');
        }
        return {
          ...control,
          image: resolved
            ? {
                image_name: resolved.imageName,
                width: resolved.width ?? (isRecord(asset) ? asset.width : undefined),
                height: resolved.height ?? (isRecord(asset) ? asset.height : undefined),
              }
            : null,
          processed_image: null,
        };
      });
      legacyMetadata = { ...metadata, control_layers: controls };
    }
    if (
      isRecord(metadata.canvas_v2_metadata) &&
      Array.isArray(metadata.canvas_v2_metadata.vectorLayers) &&
      metadata.canvas_v2_metadata.vectorLayers.length > 0
    ) {
      plan.problems.push('unsupported-legacy-layers');
    }
    let converted;
    try {
      converted = await prepareLegacyCanvasMetadata(legacyMetadata, owner, isCurrent);
    } catch (error) {
      assertAccountScopeCurrent(owner);
      if (isCurrent && !isCurrent()) {
        plan.settings = {};
        plan.problems.push('canvas-changed');
        return plan;
      }
      throw error;
    }
    assertAccountScopeCurrent(owner);
    plan.problems.push(...converted.problems);
    plan.createdImageNames = converted.createdImageNames;
    if (!converted.record) {
      return plan;
    }
    return prepareCanvasRecall({
      resolvedImages,
      metadata: { ...metadata, canvas_v3_metadata: converted.record },
      model,
      models,
      owner,
      isCurrent,
    }).then((result) => ({
      ...result,
      createdImageNames: converted.createdImageNames,
      partial: result.partial || plan.problems.length > 0,
      controlsOnly: !('canvas_v2_metadata' in metadata),
      fitControlsToFrame: standalone,
      problems: [...plan.problems, ...result.problems],
    }));
  }
  const rawDocument = record.document;
  if (!isRecord(rawDocument) || ![3, 4, 5, 6].includes(Number(rawDocument.version))) {
    plan.problems.push('invalid-document');
    return plan;
  }
  const bbox = rawDocument.bbox;
  if (
    !isRecord(bbox) ||
    ![rawDocument.width, rawDocument.height, bbox.width, bbox.height].every(
      (value) => typeof value === 'number' && Number.isSafeInteger(value) && value > 0
    ) ||
    ![bbox.x, bbox.y].every((value) => typeof value === 'number' && Number.isFinite(value))
  ) {
    plan.problems.push('invalid-document');
    return plan;
  }
  const loaded = loadCanvasState({ ...createEmptyCanvasState(), version: rawDocument.version, document: rawDocument });
  if (loaded.status !== 'loaded') {
    plan.problems.push('invalid-document');
    return plan;
  }
  if (isCurrent && !isCurrent()) {
    plan.settings = {};
    plan.problems.push('canvas-changed');
    return plan;
  }
  const document = loaded.value.document;
  // Project hydration repairs old adapter values. A metadata recall reports and skips damaged controls instead.
  const invalidControlIds = new Set<string>();
  const inspectAdapters = (nodes: unknown[], depth: number): void => {
    if (depth > 10) {
      return;
    }
    for (const raw of nodes) {
      if (!isRecord(raw)) {
        continue;
      }
      if (raw.type === 'group' && Array.isArray(raw.children)) {
        inspectAdapters(raw.children, depth + 1);
      } else if (
        raw.type === 'control' &&
        typeof raw.id === 'string' &&
        (!isRecord(raw.adapter) ||
          !isStorableControlWeight(raw.adapter.weight) ||
          !isStorableControlStepRange(raw.adapter.beginEndStepPct))
      ) {
        invalidControlIds.add(raw.id);
      }
    }
  };
  if (isRecord(rawDocument.stacks) && Array.isArray(rawDocument.stacks.control)) {
    inspectAdapters(rawDocument.stacks.control, 0);
  }
  const leaves: CanvasLayerContract[] = [];
  const collect = (nodes: CanvasNodeContract[]): void => {
    for (const node of nodes) {
      if (node.type === 'group') {
        collect(node.children);
      } else {
        leaves.push(node);
      }
    }
  };
  Object.values(document.stacks).forEach(collect);
  const names = [...new Set(leaves.flatMap(sourceNames))];
  if (!(await lookupImages(names))) {
    return plan;
  }
  const available = new Set(resolvedImages.keys());
  const controlModels = isRecord(record.controlModels) ? record.controlModels : {};
  const keep = (nodes: CanvasNodeContract[]): CanvasNodeContract[] =>
    nodes.flatMap((node): CanvasNodeContract[] => {
      if (node.type === 'group') {
        const children = keep(node.children);
        return children.length || node.children.length === 0 ? [{ ...node, children }] : [];
      }
      if (sourceNames(node).some((name) => !available.has(name))) {
        plan.problems.push(node.name);
        return [];
      }
      if (node.type === 'control' && invalidControlIds.has(node.id)) {
        plan.problems.push(node.name);
        return [];
      }
      if (node.type === 'control' && (node.adapter.model || controlModels[node.id])) {
        const resolved = resolveModel(controlModels[node.id] ?? node.adapter.model, models);
        const kind = resolved ? deriveControlKind(resolved) : null;
        if (
          !resolved ||
          resolved.base !== model.base ||
          !kind ||
          !isControlKindSupportedForBase(model.base, kind) ||
          getControlValidationReason({
            adapterModel: resolved,
            kind,
            weight: node.adapter.weight,
            beginEndStepPct: node.adapter.beginEndStepPct,
            mainBase: model.base,
            mainVariant: model.variant ?? undefined,
            controlLoraIndex: 0,
          })
        ) {
          plan.problems.push(node.name);
          return [];
        }
        node.adapter = { ...node.adapter, model: resolved.key, kind };
      }
      if (node.type === 'regional_guidance') {
        const support = getRegionalGuidanceSupport(model.base);
        if (!support && node.isEnabled) {
          plan.problems.push(node.name);
          return [];
        }
        node.referenceImages = node.referenceImages.filter((ref) => {
          const resolved = ref.config.model ? resolveModel(ref.config.model, models) : null;
          if (
            ref.config.model &&
            (!resolved ||
              resolved.base !== model.base ||
              resolved.type !== ref.config.type ||
              support?.referenceImages !== ref.config.type)
          ) {
            plan.problems.push(node.name);
            return false;
          }
          if (resolved) {
            ref.config.model = resolved;
          }
          if (ref.config.image) {
            ref.config.image.imageUrl = galleryImageUrls.full(ref.config.image.imageName);
            ref.config.image.thumbnailUrl = galleryImageUrls.thumbnail(ref.config.image.imageName);
          }
          ref.id = createItemId('regional-reference');
          return true;
        });
      }
      return [node];
    });
  for (const stack of ['raster', 'control', 'regional_guidance', 'inpaint_mask'] as const) {
    document.stacks[stack] = keep(document.stacks[stack]);
  }
  plan.partial = plan.problems.length > 0;
  if (leaves.length > 0 && Object.values(document.stacks).every((nodes) => nodes.length === 0)) {
    return plan;
  }
  plan.document = document;
  return plan;
};

export { commitCanvasRecall } from '@workbench/canvas-operations/api';

/**
 * Adopt only this preparation's pixels retained by a successful commit. Call after all synchronous Canvas/Generate
 * writes. Capture the committed name set once: subsequent project navigation must not redirect or abandon adoption.
 * Failed/unknown uploads and rejected layers remain intermediate, so server cleanup can reclaim them.
 */
export const finalizeCanvasRecallAssets = async (
  plan: CanvasRecallPlan,
  target: CanvasRecallTarget,
  owner: AccountScope
): Promise<boolean> => {
  if (!plan.createdImageNames?.length) {
    return true;
  }
  const document = target.getProject()?.canvas.document;
  if (!document || !isAccountScopeCurrent(owner)) {
    return false;
  }
  const retained = new Set<string>();
  const visit = (nodes: CanvasNodeContract[]): void => {
    for (const node of nodes) {
      if (node.type === 'group') {
        visit(node.children);
      } else {
        sourceNames(node).forEach((name) => retained.add(name));
      }
    }
  };
  Object.values(document.stacks).forEach(visit);
  const adopted = [...new Set(plan.createdImageNames)].filter((name) => retained.has(name));
  let complete = true;
  for (const imageName of adopted) {
    if (!isAccountScopeCurrent(owner)) {
      return false;
    }
    try {
      await galleryDurability.makeCanvasAsset(imageName);
    } catch {
      // The recall already applied; recent grace, open-project holds and saved references still protect these pixels.
      complete = false;
    }
  }
  return complete;
};

/** Persist pending paint before any replacement. A changed document then fails the captured-identity commit fence. */
export const settleCanvasRecall = async (
  plan: CanvasRecallPlan,
  target: CanvasRecallTarget,
  owner: AccountScope
): Promise<void> => {
  if (!plan.document) {
    return;
  }
  try {
    await getCanvasEngine(target.projectId)?.lifecycle.flushPendingUploads({ waitForHeldPixels: false });
    assertAccountScopeCurrent(owner);
  } catch (error) {
    assertAccountScopeCurrent(owner);
    plan.document = null;
    plan.settings = {};
    plan.problems.push(error instanceof Error ? error.message : 'canvas-busy');
  }
};

export const getCanvasRecallProblemsMessage = (problems: readonly string[], t: TFunction): string =>
  [...new Set(problems)]
    .map((problem) => t(`widgets.gallery.itemActions.recall.canvasReasons.${problem}`, { defaultValue: problem }))
    .join(', ');
