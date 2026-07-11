import type { BooleanRasterOperation } from '@workbench/canvas-engine/engine';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';
import type { LucideIcon } from 'lucide-react';

import { getSourceContentRect } from '@workbench/canvas-engine/document/sources';
import {
  ArrowDownIcon,
  ArrowDownToLineIcon,
  ArrowUpIcon,
  ArrowUpToLineIcon,
  CopyIcon,
  CropIcon,
  EyeIcon,
  EyeOffIcon,
  ImageIcon,
  LockIcon,
  LockOpenIcon,
  MergeIcon,
  PencilIcon,
  SaveIcon,
  ScanSearchIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';

import type { LayerMoveKind } from './layerGroups';
import type { LayerPropertiesSection } from './layerPropertiesRequestStore';

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
  | 'adjustments'
  | 'save-to-assets'
  | 'copy-to-clipboard'
  | 'crop-to-bbox'
  | 'extract-masked-area'
  | 'filter'
  | 'select-object'
  | 'run-workflow'
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

export type LayerType = CanvasLayerContract['type'];
export type LayerContextMenuSectionId = 'quick' | 'primary' | 'operations' | 'output' | 'state' | 'danger';
export type LayerContextSubmenuId = 'arrange' | 'add-modifiers' | 'add-regional' | 'boolean' | 'copy-to' | 'convert-to';

export interface LayerContextActionState {
  canRunWorkflow: boolean;
  document: CanvasDocumentContractV2;
  hasEngine: boolean;
  hasSupportedContent: boolean;
  hasWorkflowBindings: boolean;
  index: number;
  interactionLocked: boolean;
  layer: CanvasLayerContract;
}

export interface LayerContextActionEffects {
  reorder(kind: LayerMoveKind, actionId: LayerContextActionId): void;
  duplicate(): void;
  openRename(): void;
  openRunWorkflow(): void;
  startSelectObject(layerId: string): void;
  transform(): void;
  fitToBbox(): void;
  openProperties(section: LayerPropertiesSection): void;
  saveToAssets(): Promise<void>;
  copyToClipboard(): Promise<void>;
  cropToBbox(): Promise<void>;
  extractMaskedArea(): Promise<void>;
  booleanMerge(operation: BooleanRasterOperation): Promise<void>;
  copyTo(target: LayerType): void | Promise<void>;
  rasterize(): void;
  convertTo(target: LayerType): void;
  patchConfig(kind: LayerConfigPatchKind): void;
  mergeDown(): void;
  toggleVisibility(): void;
  toggleLock(): void;
  delete(): void;
}

export interface LayerContextActionRuntimeContext extends LayerContextActionState {
  effects: LayerContextActionEffects;
}

export interface LayerContextActionDefinition {
  id: LayerContextActionId;
  labelKey: string;
  defaultLabel: string;
  icon: LucideIcon;
  getIcon?(context: LayerContextActionState): LucideIcon;
  section: LayerContextMenuSectionId;
  submenu?: LayerContextSubmenuId;
  order: number;
  supportedLayerTypes: readonly LayerType[];
  tone?: 'danger';
  isVisible(context: LayerContextActionState): boolean;
  isEnabled(context: LayerContextActionState): boolean;
  handler(context: LayerContextActionRuntimeContext): void | Promise<void>;
  getDefaultLabel?(context: LayerContextActionState): string;
  getLabelKey?(context: LayerContextActionState): string;
}

export interface LayerContextAction {
  id: LayerContextActionId;
  labelKey: string;
  defaultLabel: string;
  icon: LucideIcon;
  section: LayerContextMenuSectionId;
  submenu?: LayerContextSubmenuId;
  order: number;
  tone?: 'danger';
  isDisabled: boolean;
  handler(context: LayerContextActionRuntimeContext): void | Promise<void>;
}

export type LayerConfigPatchKind =
  | 'control-transparency-effect'
  | 'regional-positive-prompt'
  | 'regional-negative-prompt'
  | 'regional-reference-image'
  | 'regional-auto-negative'
  | 'inpaint-noise'
  | 'inpaint-denoise-limit';

const ALL_LAYER_TYPES = ['raster', 'control', 'inpaint_mask', 'regional_guidance'] as const;
const RASTER_ONLY = ['raster'] as const;
const CONTROL_ONLY = ['control'] as const;
const RASTER_AND_CONTROL = ['raster', 'control'] as const;
const INPAINT_ONLY = ['inpaint_mask'] as const;
const REGIONAL_ONLY = ['regional_guidance'] as const;

const alwaysVisible = (): boolean => true;
const isInteractionFree = (context: LayerContextActionState): boolean => !context.interactionLocked;
const isLayerMutable = (context: LayerContextActionState): boolean =>
  isInteractionFree(context) && !context.layer.isLocked;
const hasReadablePixels = (context: LayerContextActionState): boolean =>
  context.hasEngine && context.hasSupportedContent && !context.interactionLocked;
const hasMutablePixels = (context: LayerContextActionState): boolean =>
  isLayerMutable(context) && hasReadablePixels(context);
const hasTransformablePixels = (context: LayerContextActionState): boolean =>
  context.layer.isEnabled && hasMutablePixels(context);
const hasDocumentContent = (context: LayerContextActionState): boolean => {
  const contentRect = getSourceContentRect(context.layer, context.document);
  return contentRect.width > 0 && contentRect.height > 0;
};

const isParametricRasterizable = (layer: CanvasLayerContract): boolean =>
  layer.type === 'raster' &&
  (layer.source.type === 'gradient' ||
    layer.source.type === 'text' ||
    (layer.source.type === 'shape' && layer.source.kind !== 'polygon'));

const isPixelBacked = (layer: CanvasLayerContract): boolean =>
  (layer.type === 'raster' || layer.type === 'control') &&
  (layer.source.type === 'image' || layer.source.type === 'paint');

const hasFilterableLayerContent = (context: LayerContextActionState): boolean => {
  if (!context.hasSupportedContent || (context.layer.type !== 'raster' && context.layer.type !== 'control')) {
    return false;
  }
  const { source } = context.layer;
  if (source.type === 'image' || source.type === 'paint') {
    // `hasSupportedContent` distinguishes empty paint from live unpersisted
    // cache pixels, which are intentionally filterable despite `bitmap: null`.
    return true;
  }
  if (context.layer.type !== 'raster') {
    return false;
  }
  return (
    source.type === 'text' ||
    source.type === 'gradient' ||
    (source.type === 'shape' && (source.kind === 'rect' || source.kind === 'ellipse'))
  );
};

const groupPosition = (context: LayerContextActionState) => getGroupPosition(context.document.layers, context.layer.id);

const canMoveForward = (context: LayerContextActionState): boolean => {
  const position = groupPosition(context);
  return isInteractionFree(context) && !!position && position.index > 0;
};

const canMoveBackward = (context: LayerContextActionState): boolean => {
  const position = groupPosition(context);
  return isInteractionFree(context) && !!position && position.index < position.count - 1;
};

const hasMergeableLayerBelow = (context: LayerContextActionState): boolean =>
  canMergeLayerDown(context.document.layers, context.index, true);

const isBooleanRasterLayer = (layer: CanvasLayerContract | undefined): boolean =>
  !!layer &&
  layer.isEnabled &&
  layer.type === 'raster' &&
  (layer.source.type === 'paint' || layer.source.type === 'image');

const hasBooleanRasterPair = (context: LayerContextActionState): boolean =>
  isBooleanRasterLayer(context.document.layers[context.index]) &&
  isBooleanRasterLayer(context.document.layers[context.index + 1]);

export const LAYER_CONTEXT_ACTION_DEFINITIONS: readonly LayerContextActionDefinition[] = [
  {
    defaultLabel: 'Move to front',
    handler: ({ effects }) => effects.reorder('front', 'move-to-front'),
    icon: ArrowUpToLineIcon,
    id: 'move-to-front',
    isEnabled: canMoveForward,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.moveToFront',
    order: 0,
    section: 'quick',
    submenu: 'arrange',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Move forward',
    handler: ({ effects }) => effects.reorder('forward', 'move-forward'),
    icon: ArrowUpIcon,
    id: 'move-forward',
    isEnabled: canMoveForward,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.moveForward',
    order: 1,
    section: 'quick',
    submenu: 'arrange',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Move backward',
    handler: ({ effects }) => effects.reorder('backward', 'move-backward'),
    icon: ArrowDownIcon,
    id: 'move-backward',
    isEnabled: canMoveBackward,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.moveBackward',
    order: 2,
    section: 'quick',
    submenu: 'arrange',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Move to back',
    handler: ({ effects }) => effects.reorder('back', 'move-to-back'),
    icon: ArrowDownToLineIcon,
    id: 'move-to-back',
    isEnabled: canMoveBackward,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.moveToBack',
    order: 3,
    section: 'quick',
    submenu: 'arrange',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Duplicate',
    handler: ({ effects }) => effects.duplicate(),
    icon: CopyIcon,
    id: 'duplicate',
    isEnabled: isInteractionFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.duplicate',
    order: 10,
    section: 'quick',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Add image noise',
    handler: ({ effects }) => effects.patchConfig('inpaint-noise'),
    icon: SlidersHorizontalIcon,
    id: 'inpaint-noise',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'inpaint_mask' && context.layer.noiseLevel === undefined,
    labelKey: 'widgets.layers.actions.addImageNoise',
    order: 0,
    section: 'primary',
    submenu: 'add-modifiers',
    supportedLayerTypes: INPAINT_ONLY,
  },
  {
    defaultLabel: 'Add denoise limit',
    handler: ({ effects }) => effects.patchConfig('inpaint-denoise-limit'),
    icon: SlidersHorizontalIcon,
    id: 'inpaint-denoise-limit',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'inpaint_mask' && context.layer.denoiseLimit === undefined,
    labelKey: 'widgets.layers.actions.addDenoiseLimit',
    order: 1,
    section: 'primary',
    submenu: 'add-modifiers',
    supportedLayerTypes: INPAINT_ONLY,
  },
  {
    defaultLabel: 'Add positive prompt',
    handler: ({ effects }) => effects.patchConfig('regional-positive-prompt'),
    icon: PencilIcon,
    id: 'regional-positive-prompt',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'regional_guidance' && context.layer.positivePrompt === null,
    labelKey: 'widgets.layers.actions.addPositivePrompt',
    order: 0,
    section: 'primary',
    submenu: 'add-regional',
    supportedLayerTypes: REGIONAL_ONLY,
  },
  {
    defaultLabel: 'Add negative prompt',
    handler: ({ effects }) => effects.patchConfig('regional-negative-prompt'),
    icon: PencilIcon,
    id: 'regional-negative-prompt',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'regional_guidance' && context.layer.negativePrompt === null,
    labelKey: 'widgets.layers.actions.addNegativePrompt',
    order: 1,
    section: 'primary',
    submenu: 'add-regional',
    supportedLayerTypes: REGIONAL_ONLY,
  },
  {
    defaultLabel: 'Add reference image',
    handler: ({ effects }) => effects.patchConfig('regional-reference-image'),
    icon: ImageIcon,
    id: 'regional-reference-image',
    isEnabled: isLayerMutable,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.addReferenceImage',
    order: 2,
    section: 'primary',
    submenu: 'add-regional',
    supportedLayerTypes: REGIONAL_ONLY,
  },
  {
    defaultLabel: 'Transform',
    handler: ({ effects }) => effects.transform(),
    icon: SlidersHorizontalIcon,
    id: 'transform',
    isEnabled: hasTransformablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.transform',
    order: 10,
    section: 'primary',
    supportedLayerTypes: ['raster', 'control'],
  },
  {
    defaultLabel: 'Rename',
    handler: ({ effects }) => effects.openRename(),
    icon: PencilIcon,
    id: 'rename',
    isEnabled: isInteractionFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.rename',
    order: 20,
    section: 'primary',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Fit to bbox',
    handler: ({ effects }) => effects.fitToBbox(),
    icon: ImageIcon,
    id: 'fit-to-bbox',
    isEnabled: (context) => hasDocumentContent(context) && isLayerMutable(context),
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.fitToBbox',
    order: 30,
    section: 'primary',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Adjustments',
    handler: ({ effects }) => effects.openProperties('adjustments'),
    icon: SlidersHorizontalIcon,
    id: 'adjustments',
    isEnabled: hasMutablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.adjustments',
    order: 31,
    section: 'primary',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Filter',
    handler: ({ effects }) => effects.openProperties('filter'),
    icon: SlidersHorizontalIcon,
    id: 'filter',
    isEnabled: hasMutablePixels,
    isVisible: hasFilterableLayerContent,
    labelKey: 'widgets.layers.control.filter',
    order: 40,
    section: 'primary',
    supportedLayerTypes: RASTER_AND_CONTROL,
  },
  {
    defaultLabel: 'Select object',
    handler: ({ effects, layer }) => effects.startSelectObject(layer.id),
    icon: ScanSearchIcon,
    id: 'select-object',
    isEnabled: hasMutablePixels,
    isVisible: (context) => context.hasSupportedContent,
    labelKey: 'widgets.layers.actions.selectObject',
    order: 41,
    section: 'primary',
    supportedLayerTypes: RASTER_AND_CONTROL,
  },
  {
    defaultLabel: 'Run workflow',
    handler: ({ effects }) => effects.openRunWorkflow(),
    icon: WorkflowIcon,
    id: 'run-workflow',
    isEnabled: (context) => context.canRunWorkflow && hasMutablePixels(context),
    isVisible: (context) => context.hasWorkflowBindings && context.hasSupportedContent,
    labelKey: 'widgets.layers.actions.runWorkflow',
    order: 42,
    section: 'primary',
    supportedLayerTypes: RASTER_AND_CONTROL,
  },
  {
    defaultLabel: 'Toggle transparency effect',
    getDefaultLabel: (context) =>
      context.layer.type === 'control' && context.layer.withTransparencyEffect
        ? 'Disable transparency effect'
        : 'Enable transparency effect',
    getLabelKey: (context) =>
      context.layer.type === 'control' && context.layer.withTransparencyEffect
        ? 'widgets.layers.actions.disableTransparencyEffect'
        : 'widgets.layers.actions.enableTransparencyEffect',
    handler: ({ effects }) => effects.patchConfig('control-transparency-effect'),
    icon: SlidersHorizontalIcon,
    id: 'control-transparency-effect',
    isEnabled: isLayerMutable,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleTransparencyEffect',
    order: 50,
    section: 'primary',
    supportedLayerTypes: CONTROL_ONLY,
  },
  {
    defaultLabel: 'Toggle auto-negative',
    getDefaultLabel: (context) =>
      context.layer.type === 'regional_guidance' && context.layer.autoNegative
        ? 'Disable auto-negative'
        : 'Enable auto-negative',
    getLabelKey: (context) =>
      context.layer.type === 'regional_guidance' && context.layer.autoNegative
        ? 'widgets.layers.actions.disableAutoNegative'
        : 'widgets.layers.actions.enableAutoNegative',
    handler: ({ effects }) => effects.patchConfig('regional-auto-negative'),
    icon: SlidersHorizontalIcon,
    id: 'regional-auto-negative',
    isEnabled: isLayerMutable,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleAutoNegative',
    order: 50,
    section: 'primary',
    supportedLayerTypes: REGIONAL_ONLY,
  },
  {
    defaultLabel: 'Extract masked area',
    handler: ({ effects }) => effects.extractMaskedArea(),
    icon: CropIcon,
    id: 'extract-masked-area',
    isEnabled: hasMutablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.extractMaskedArea',
    order: 60,
    section: 'primary',
    supportedLayerTypes: INPAINT_ONLY,
  },
  {
    defaultLabel: 'Merge down',
    handler: ({ effects }) => effects.mergeDown(),
    icon: MergeIcon,
    id: 'merge-down',
    isEnabled: (context) => hasMutablePixels(context) && hasMergeableLayerBelow(context),
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.mergeDown',
    order: 0,
    section: 'operations',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Intersect with layer below',
    handler: ({ effects }) => effects.booleanMerge('intersect'),
    icon: MergeIcon,
    id: 'intersect',
    isEnabled: (context) => hasMutablePixels(context) && hasMergeableLayerBelow(context),
    isVisible: hasBooleanRasterPair,
    labelKey: 'widgets.layers.actions.intersect',
    order: 10,
    section: 'operations',
    submenu: 'boolean',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Cutout with layer below',
    handler: ({ effects }) => effects.booleanMerge('cutout'),
    icon: MergeIcon,
    id: 'cutout',
    isEnabled: (context) => hasMutablePixels(context) && hasMergeableLayerBelow(context),
    isVisible: hasBooleanRasterPair,
    labelKey: 'widgets.layers.actions.cutout',
    order: 11,
    section: 'operations',
    submenu: 'boolean',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Cutaway with layer below',
    handler: ({ effects }) => effects.booleanMerge('cutaway'),
    icon: MergeIcon,
    id: 'cutaway',
    isEnabled: (context) => hasMutablePixels(context) && hasMergeableLayerBelow(context),
    isVisible: hasBooleanRasterPair,
    labelKey: 'widgets.layers.actions.cutaway',
    order: 12,
    section: 'operations',
    submenu: 'boolean',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Exclude layer below',
    handler: ({ effects }) => effects.booleanMerge('exclude'),
    icon: MergeIcon,
    id: 'exclude',
    isEnabled: (context) => hasMutablePixels(context) && hasMergeableLayerBelow(context),
    isVisible: hasBooleanRasterPair,
    labelKey: 'widgets.layers.actions.exclude',
    order: 13,
    section: 'operations',
    submenu: 'boolean',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Copy layer to clipboard',
    handler: ({ effects }) => effects.copyToClipboard(),
    icon: CopyIcon,
    id: 'copy-to-clipboard',
    isEnabled: hasReadablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.copyLayerToClipboard',
    order: 20,
    section: 'operations',
    submenu: 'copy-to',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Copy to raster layer',
    handler: ({ effects }) => effects.copyTo('raster'),
    icon: CopyIcon,
    id: 'copy-to-raster',
    isEnabled: hasReadablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.copyToRasterLayer',
    order: 21,
    section: 'operations',
    submenu: 'copy-to',
    supportedLayerTypes: ['control', 'inpaint_mask', 'regional_guidance'],
  },
  {
    defaultLabel: 'Copy to control layer',
    handler: ({ effects }) => effects.copyTo('control'),
    icon: CopyIcon,
    id: 'copy-to-control',
    isEnabled: hasReadablePixels,
    isVisible: (context) => isPixelBacked(context.layer),
    labelKey: 'widgets.layers.actions.copyToControl',
    order: 22,
    section: 'operations',
    submenu: 'copy-to',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Copy to inpaint mask',
    handler: ({ effects }) => effects.copyTo('inpaint_mask'),
    icon: CopyIcon,
    id: 'copy-to-inpaint-mask',
    isEnabled: hasReadablePixels,
    isVisible: (context) => isPixelBacked(context.layer) || context.layer.type === 'regional_guidance',
    labelKey: 'widgets.layers.actions.copyToInpaintMask',
    order: 23,
    section: 'operations',
    submenu: 'copy-to',
    supportedLayerTypes: ['raster', 'control', 'regional_guidance'],
  },
  {
    defaultLabel: 'Copy to regional guidance',
    handler: ({ effects }) => effects.copyTo('regional_guidance'),
    icon: CopyIcon,
    id: 'copy-to-regional-guidance',
    isEnabled: hasReadablePixels,
    isVisible: (context) => isPixelBacked(context.layer) || context.layer.type === 'inpaint_mask',
    labelKey: 'widgets.layers.actions.copyToRegionalGuidance',
    order: 24,
    section: 'operations',
    submenu: 'copy-to',
    supportedLayerTypes: ['raster', 'control', 'inpaint_mask'],
  },
  {
    defaultLabel: 'Rasterize',
    handler: ({ effects }) => effects.rasterize(),
    icon: ImageIcon,
    id: 'rasterize',
    isEnabled: hasMutablePixels,
    isVisible: (context) => isParametricRasterizable(context.layer),
    labelKey: 'widgets.layers.actions.rasterize',
    order: 30,
    section: 'operations',
    submenu: 'convert-to',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Convert to Control Layer',
    handler: ({ effects }) => effects.convertTo('control'),
    icon: SlidersHorizontalIcon,
    id: 'convert-to-control',
    isEnabled: isLayerMutable,
    isVisible: (context) => canConvertRasterControl(context.layer),
    labelKey: 'widgets.layers.actions.convertToControl',
    order: 31,
    section: 'operations',
    submenu: 'convert-to',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Convert to Raster Layer',
    handler: ({ effects }) => effects.convertTo('raster'),
    icon: ImageIcon,
    id: 'convert-to-raster',
    isEnabled: isLayerMutable,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.convertToRaster',
    order: 32,
    section: 'operations',
    submenu: 'convert-to',
    supportedLayerTypes: CONTROL_ONLY,
  },
  {
    defaultLabel: 'Convert to inpaint mask',
    handler: ({ effects }) => effects.convertTo('inpaint_mask'),
    icon: ImageIcon,
    id: 'convert-to-inpaint-mask',
    isEnabled: isLayerMutable,
    isVisible: (context) => isPixelBacked(context.layer),
    labelKey: 'widgets.layers.actions.convertToInpaintMask',
    order: 33,
    section: 'operations',
    submenu: 'convert-to',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Convert to regional guidance',
    handler: ({ effects }) => effects.convertTo('regional_guidance'),
    icon: ImageIcon,
    id: 'convert-to-regional-guidance',
    isEnabled: isLayerMutable,
    isVisible: (context) => isPixelBacked(context.layer),
    labelKey: 'widgets.layers.actions.convertToRegionalGuidance',
    order: 34,
    section: 'operations',
    submenu: 'convert-to',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Crop layer to bbox',
    handler: ({ effects }) => effects.cropToBbox(),
    icon: CropIcon,
    id: 'crop-to-bbox',
    isEnabled: hasMutablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.cropLayerToBbox',
    order: 0,
    section: 'output',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Save layer to assets',
    handler: ({ effects }) => effects.saveToAssets(),
    icon: SaveIcon,
    id: 'save-to-assets',
    isEnabled: hasReadablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.saveLayerToAssets',
    order: 10,
    section: 'output',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Toggle visibility',
    getDefaultLabel: (context) => (context.layer.isEnabled ? 'Hide' : 'Show'),
    getIcon: (context) => (context.layer.isEnabled ? EyeOffIcon : EyeIcon),
    getLabelKey: (context) => (context.layer.isEnabled ? 'widgets.layers.actions.hide' : 'widgets.layers.actions.show'),
    handler: ({ effects }) => effects.toggleVisibility(),
    icon: EyeIcon,
    id: 'toggle-visibility',
    isEnabled: isInteractionFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleVisibility',
    order: 0,
    section: 'state',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Toggle lock',
    getDefaultLabel: (context) => (context.layer.isLocked ? 'Unlock' : 'Lock'),
    getIcon: (context) => (context.layer.isLocked ? LockOpenIcon : LockIcon),
    getLabelKey: (context) =>
      context.layer.isLocked ? 'widgets.layers.actions.unlock' : 'widgets.layers.actions.lock',
    handler: ({ effects }) => effects.toggleLock(),
    icon: LockIcon,
    id: 'toggle-lock',
    isEnabled: isInteractionFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleLock',
    order: 10,
    section: 'state',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Delete',
    handler: ({ effects }) => effects.delete(),
    icon: Trash2Icon,
    id: 'delete',
    isEnabled: isLayerMutable,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.delete',
    order: 0,
    section: 'danger',
    supportedLayerTypes: ALL_LAYER_TYPES,
    tone: 'danger',
  },
];

export const getLayerContextActionDefinition = (id: LayerContextActionId): LayerContextActionDefinition => {
  const definition = LAYER_CONTEXT_ACTION_DEFINITIONS.find((candidate) => candidate.id === id);
  if (!definition) {
    throw new Error(`Unknown layer context action: ${id}`);
  }
  return definition;
};

export const getLayerContextActions = (context: LayerContextActionState): LayerContextAction[] =>
  LAYER_CONTEXT_ACTION_DEFINITIONS.filter(
    (definition) => definition.supportedLayerTypes.includes(context.layer.type) && definition.isVisible(context)
  ).map((definition) => ({
    defaultLabel: definition.getDefaultLabel?.(context) ?? definition.defaultLabel,
    handler: definition.handler,
    icon: definition.getIcon?.(context) ?? definition.icon,
    id: definition.id,
    isDisabled: !definition.isEnabled(context),
    labelKey: definition.getLabelKey?.(context) ?? definition.labelKey,
    order: definition.order,
    section: definition.section,
    submenu: definition.submenu,
    tone: definition.tone,
  }));
