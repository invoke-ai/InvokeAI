import type { CanvasLayerContract } from '@workbench/types';

import { getGroupPosition } from './layerGroups';
import { canConvertRasterControl, canMergeLayerDown } from './layerOps';

export type LayerContextActionId =
  | 'move-to-front'
  | 'move-forward'
  | 'move-backward'
  | 'move-to-back'
  | 'duplicate'
  | 'rename'
  | 'transform'
  | 'fit-to-bbox'
  | 'save-to-assets'
  | 'copy-to-clipboard'
  | 'crop-to-bbox'
  | 'extract-masked-area'
  | 'intersect'
  | 'cutout'
  | 'cutaway'
  | 'exclude'
  | 'copy-to-raster'
  | 'copy-to-control'
  | 'copy-to-inpaint-mask'
  | 'copy-to-regional-guidance'
  | 'rasterize'
  | 'convert-to-control'
  | 'convert-to-raster'
  | 'convert-to-inpaint-mask'
  | 'convert-to-regional-guidance'
  | 'control-transparency-effect'
  | 'regional-positive-prompt'
  | 'regional-negative-prompt'
  | 'regional-reference-image'
  | 'regional-auto-negative'
  | 'inpaint-noise'
  | 'inpaint-denoise-limit'
  | 'merge-down'
  | 'toggle-visibility'
  | 'toggle-lock'
  | 'delete';

export type LayerContextActionGroup = 'icons' | 'edit' | 'convert' | 'layerConfig' | 'state' | 'danger';

export interface LayerContextActionContext {
  hasEngine: boolean;
  index: number;
  layer: CanvasLayerContract;
  layers: readonly CanvasLayerContract[];
}

export interface LayerContextAction {
  defaultLabel: string;
  group: LayerContextActionGroup;
  id: LayerContextActionId;
  isDisabled: boolean;
  labelKey: string;
  tone?: 'danger';
}

type LayerContextActionDefinition = Omit<LayerContextAction, 'isDisabled'> & {
  getDefaultLabel?: (ctx: LayerContextActionContext) => string;
  getLabelKey?: (ctx: LayerContextActionContext) => string;
  isDisabled?: (ctx: LayerContextActionContext) => boolean;
  isVisible?: (ctx: LayerContextActionContext) => boolean;
};

const isParametricRasterizable = (layer: CanvasLayerContract): boolean =>
  layer.type === 'raster' &&
  (layer.source.type === 'gradient' ||
    layer.source.type === 'text' ||
    (layer.source.type === 'shape' && layer.source.kind !== 'polygon'));

const isPixelBacked = (layer: CanvasLayerContract): boolean =>
  (layer.type === 'raster' || layer.type === 'control') &&
  (layer.source.type === 'image' || layer.source.type === 'paint');

const groupPosition = (ctx: LayerContextActionContext) => getGroupPosition(ctx.layers, ctx.layer.id);

const canMoveForward = (ctx: LayerContextActionContext): boolean => {
  const position = groupPosition(ctx);
  return !!position && position.index > 0;
};

const canMoveBackward = (ctx: LayerContextActionContext): boolean => {
  const position = groupPosition(ctx);
  return !!position && position.index < position.count - 1;
};

export const LAYER_CONTEXT_ACTION_DEFINITIONS: readonly LayerContextActionDefinition[] = [
  {
    defaultLabel: 'Move to front',
    group: 'icons',
    id: 'move-to-front',
    isDisabled: (ctx) => !canMoveForward(ctx),
    labelKey: 'widgets.layers.actions.moveToFront',
  },
  {
    defaultLabel: 'Move forward',
    group: 'icons',
    id: 'move-forward',
    isDisabled: (ctx) => !canMoveForward(ctx),
    labelKey: 'widgets.layers.actions.moveForward',
  },
  {
    defaultLabel: 'Move backward',
    group: 'icons',
    id: 'move-backward',
    isDisabled: (ctx) => !canMoveBackward(ctx),
    labelKey: 'widgets.layers.actions.moveBackward',
  },
  {
    defaultLabel: 'Move to back',
    group: 'icons',
    id: 'move-to-back',
    isDisabled: (ctx) => !canMoveBackward(ctx),
    labelKey: 'widgets.layers.actions.moveToBack',
  },
  { defaultLabel: 'Duplicate', group: 'icons', id: 'duplicate', labelKey: 'widgets.layers.actions.duplicate' },
  { defaultLabel: 'Rename', group: 'edit', id: 'rename', labelKey: 'widgets.layers.actions.rename' },
  {
    defaultLabel: 'Transform',
    group: 'edit',
    id: 'transform',
    isDisabled: (ctx) => !ctx.hasEngine,
    labelKey: 'widgets.layers.actions.transform',
  },
  { defaultLabel: 'Fit to bbox', group: 'edit', id: 'fit-to-bbox', labelKey: 'widgets.layers.actions.fitToBbox' },
  {
    defaultLabel: 'Save layer to assets',
    group: 'edit',
    id: 'save-to-assets',
    isDisabled: (ctx) => !ctx.hasEngine,
    labelKey: 'widgets.layers.actions.saveLayerToAssets',
  },
  {
    defaultLabel: 'Copy layer to clipboard',
    group: 'edit',
    id: 'copy-to-clipboard',
    isDisabled: (ctx) => !ctx.hasEngine,
    labelKey: 'widgets.layers.actions.copyLayerToClipboard',
  },
  {
    defaultLabel: 'Crop layer to bbox',
    group: 'edit',
    id: 'crop-to-bbox',
    isDisabled: (ctx) => !ctx.hasEngine || ctx.layer.isLocked,
    labelKey: 'widgets.layers.actions.cropLayerToBbox',
  },
  {
    defaultLabel: 'Extract masked area',
    group: 'edit',
    id: 'extract-masked-area',
    isDisabled: (ctx) => !ctx.hasEngine,
    isVisible: (ctx) => ctx.layer.type === 'inpaint_mask',
    labelKey: 'widgets.layers.actions.extractMaskedArea',
  },
  {
    defaultLabel: 'Intersect with layer below',
    group: 'edit',
    id: 'intersect',
    isVisible: (ctx) => canMergeLayerDown(ctx.layers, ctx.index, ctx.hasEngine),
    labelKey: 'widgets.layers.actions.intersect',
  },
  {
    defaultLabel: 'Cutout with layer below',
    group: 'edit',
    id: 'cutout',
    isVisible: (ctx) => canMergeLayerDown(ctx.layers, ctx.index, ctx.hasEngine),
    labelKey: 'widgets.layers.actions.cutout',
  },
  {
    defaultLabel: 'Cutaway with layer below',
    group: 'edit',
    id: 'cutaway',
    isVisible: (ctx) => canMergeLayerDown(ctx.layers, ctx.index, ctx.hasEngine),
    labelKey: 'widgets.layers.actions.cutaway',
  },
  {
    defaultLabel: 'Exclude layer below',
    group: 'edit',
    id: 'exclude',
    isVisible: (ctx) => canMergeLayerDown(ctx.layers, ctx.index, ctx.hasEngine),
    labelKey: 'widgets.layers.actions.exclude',
  },
  {
    defaultLabel: 'Copy to raster layer',
    group: 'convert',
    id: 'copy-to-raster',
    isDisabled: (ctx) => !ctx.hasEngine,
    isVisible: (ctx) => ctx.layer.type !== 'raster',
    labelKey: 'widgets.layers.actions.copyToRasterLayer',
  },
  {
    defaultLabel: 'Copy to control layer',
    group: 'convert',
    id: 'copy-to-control',
    isDisabled: (ctx) => !ctx.hasEngine,
    isVisible: (ctx) => ctx.layer.type === 'raster' && isPixelBacked(ctx.layer),
    labelKey: 'widgets.layers.actions.copyToControl',
  },
  {
    defaultLabel: 'Copy to inpaint mask',
    group: 'convert',
    id: 'copy-to-inpaint-mask',
    isDisabled: (ctx) => !ctx.hasEngine,
    isVisible: (ctx) => isPixelBacked(ctx.layer) || ctx.layer.type === 'regional_guidance',
    labelKey: 'widgets.layers.actions.copyToInpaintMask',
  },
  {
    defaultLabel: 'Copy to regional guidance',
    group: 'convert',
    id: 'copy-to-regional-guidance',
    isDisabled: (ctx) => !ctx.hasEngine,
    isVisible: (ctx) => isPixelBacked(ctx.layer) || ctx.layer.type === 'inpaint_mask',
    labelKey: 'widgets.layers.actions.copyToRegionalGuidance',
  },
  {
    defaultLabel: 'Rasterize',
    group: 'convert',
    id: 'rasterize',
    isVisible: (ctx) => ctx.hasEngine && isParametricRasterizable(ctx.layer),
    labelKey: 'widgets.layers.actions.rasterize',
  },
  {
    defaultLabel: 'Convert to Control Layer',
    group: 'convert',
    id: 'convert-to-control',
    isVisible: (ctx) => ctx.layer.type === 'raster' && canConvertRasterControl(ctx.layer),
    labelKey: 'widgets.layers.actions.convertToControl',
  },
  {
    defaultLabel: 'Convert to Raster Layer',
    group: 'convert',
    id: 'convert-to-raster',
    isVisible: (ctx) => ctx.layer.type === 'control',
    labelKey: 'widgets.layers.actions.convertToRaster',
  },
  {
    defaultLabel: 'Convert to inpaint mask',
    group: 'convert',
    id: 'convert-to-inpaint-mask',
    isVisible: (ctx) => ctx.layer.type === 'raster' && isPixelBacked(ctx.layer),
    labelKey: 'widgets.layers.actions.convertToInpaintMask',
  },
  {
    defaultLabel: 'Convert to regional guidance',
    group: 'convert',
    id: 'convert-to-regional-guidance',
    isVisible: (ctx) => ctx.layer.type === 'raster' && isPixelBacked(ctx.layer),
    labelKey: 'widgets.layers.actions.convertToRegionalGuidance',
  },
  {
    getDefaultLabel: (ctx) =>
      ctx.layer.type === 'control' && ctx.layer.withTransparencyEffect
        ? 'Disable transparency effect'
        : 'Enable transparency effect',
    getLabelKey: (ctx) =>
      ctx.layer.type === 'control' && ctx.layer.withTransparencyEffect
        ? 'widgets.layers.actions.disableTransparencyEffect'
        : 'widgets.layers.actions.enableTransparencyEffect',
    defaultLabel: 'Toggle transparency effect',
    group: 'layerConfig',
    id: 'control-transparency-effect',
    isVisible: (ctx) => ctx.layer.type === 'control',
    labelKey: 'widgets.layers.actions.toggleTransparencyEffect',
  },
  {
    defaultLabel: 'Add positive prompt',
    group: 'layerConfig',
    id: 'regional-positive-prompt',
    isVisible: (ctx) => ctx.layer.type === 'regional_guidance' && ctx.layer.positivePrompt === null,
    labelKey: 'widgets.layers.actions.addPositivePrompt',
  },
  {
    defaultLabel: 'Add negative prompt',
    group: 'layerConfig',
    id: 'regional-negative-prompt',
    isVisible: (ctx) => ctx.layer.type === 'regional_guidance' && ctx.layer.negativePrompt === null,
    labelKey: 'widgets.layers.actions.addNegativePrompt',
  },
  {
    defaultLabel: 'Add reference image',
    group: 'layerConfig',
    id: 'regional-reference-image',
    isVisible: (ctx) => ctx.layer.type === 'regional_guidance',
    labelKey: 'widgets.layers.actions.addReferenceImage',
  },
  {
    getDefaultLabel: (ctx) =>
      ctx.layer.type === 'regional_guidance' && ctx.layer.autoNegative
        ? 'Disable auto-negative'
        : 'Enable auto-negative',
    getLabelKey: (ctx) =>
      ctx.layer.type === 'regional_guidance' && ctx.layer.autoNegative
        ? 'widgets.layers.actions.disableAutoNegative'
        : 'widgets.layers.actions.enableAutoNegative',
    defaultLabel: 'Toggle auto-negative',
    group: 'layerConfig',
    id: 'regional-auto-negative',
    isVisible: (ctx) => ctx.layer.type === 'regional_guidance',
    labelKey: 'widgets.layers.actions.toggleAutoNegative',
  },
  {
    defaultLabel: 'Add image noise',
    group: 'layerConfig',
    id: 'inpaint-noise',
    isVisible: (ctx) => ctx.layer.type === 'inpaint_mask' && ctx.layer.noiseLevel === undefined,
    labelKey: 'widgets.layers.actions.addImageNoise',
  },
  {
    defaultLabel: 'Add denoise limit',
    group: 'layerConfig',
    id: 'inpaint-denoise-limit',
    isVisible: (ctx) => ctx.layer.type === 'inpaint_mask' && ctx.layer.denoiseLimit === undefined,
    labelKey: 'widgets.layers.actions.addDenoiseLimit',
  },
  {
    defaultLabel: 'Merge down',
    group: 'state',
    id: 'merge-down',
    isDisabled: (ctx) => !canMergeLayerDown(ctx.layers, ctx.index, ctx.hasEngine),
    labelKey: 'widgets.layers.actions.mergeDown',
  },
  {
    getDefaultLabel: (ctx) => (ctx.layer.isEnabled ? 'Hide' : 'Show'),
    getLabelKey: (ctx) => (ctx.layer.isEnabled ? 'widgets.layers.actions.hide' : 'widgets.layers.actions.show'),
    defaultLabel: 'Toggle visibility',
    group: 'state',
    id: 'toggle-visibility',
    labelKey: 'widgets.layers.actions.toggleVisibility',
  },
  {
    getDefaultLabel: (ctx) => (ctx.layer.isLocked ? 'Unlock' : 'Lock'),
    getLabelKey: (ctx) => (ctx.layer.isLocked ? 'widgets.layers.actions.unlock' : 'widgets.layers.actions.lock'),
    defaultLabel: 'Toggle lock',
    group: 'state',
    id: 'toggle-lock',
    labelKey: 'widgets.layers.actions.toggleLock',
  },
  { defaultLabel: 'Delete', group: 'danger', id: 'delete', labelKey: 'widgets.layers.actions.delete', tone: 'danger' },
];

export const getLayerContextActions = (ctx: LayerContextActionContext): LayerContextAction[] =>
  LAYER_CONTEXT_ACTION_DEFINITIONS.filter((definition) => definition.isVisible?.(ctx) ?? true).map((definition) => ({
    defaultLabel: definition.getDefaultLabel?.(ctx) ?? definition.defaultLabel,
    group: definition.group,
    id: definition.id,
    isDisabled: definition.isDisabled?.(ctx) ?? false,
    labelKey: definition.getLabelKey?.(ctx) ?? definition.labelKey,
    tone: definition.tone,
  }));
