import type {
  BooleanRasterOperation,
  CanvasAdjustmentEntry,
  CanvasColorLabel,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  LayerStackMoveKind,
} from '@workbench/canvas-engine/api';
import type { LucideIcon } from 'lucide-react';

import {
  getDocumentIndex,
  getSourceContentRect,
  isEmptyPolygonShape,
  isHideableLayer,
  isNodeHidden,
  isPixelBackedLayer,
  lookupLayerBelow,
} from '@workbench/canvas-engine/api';
import {
  ApertureIcon,
  WandSparklesIcon,
  ArrowDownIcon,
  ArrowDownToLineIcon,
  ArrowUpIcon,
  ArrowUpToLineIcon,
  CircleIcon,
  CircleOffIcon,
  CopyIcon,
  CropIcon,
  DropletIcon,
  EyeIcon,
  EyeOffIcon,
  FolderPlusIcon,
  GaugeIcon,
  ImageIcon,
  ImagePlusIcon,
  LockIcon,
  LockOpenIcon,
  MergeIcon,
  PencilIcon,
  SaveIcon,
  ContrastIcon,
  RainbowIcon,
  ScanSearchIcon,
  SlidersVerticalIcon,
  SplineIcon,
  SunMediumIcon,
  WavesIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';

import { COLOR_LABEL_ITEMS } from './colorLabels';
import { canAddRegionalReferenceImage, canConvertRasterControl, canMergeLayerDown } from './layerOps';

export type LayerContextActionId =
  | 'add-reference-image'
  | 'add-noise'
  | 'add-regenerate-region'
  | 'add-denoise-limit'
  | 'add-brightness-contrast'
  | 'add-exposure'
  | 'add-levels'
  | 'add-curves'
  | 'add-hsl'
  | 'add-hue'
  | 'add-invert'
  | 'merge-selected'
  | 'move-to-front'
  | 'move-forward'
  | 'move-backward'
  | 'move-to-back'
  | 'duplicate'
  | 'group'
  | 'rename'
  | 'transform'
  | 'fit-to-bbox'
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
  | 'regional-auto-negative'
  | 'merge-down'
  | 'toggle-visibility'
  | 'toggle-hidden'
  | 'toggle-lock'
  | 'color-label-none'
  | `color-label-${CanvasColorLabel}`
  | 'delete';

export type LayerType = CanvasLayerContract['type'];
export type LayerContextMenuSectionId = 'quick' | 'primary' | 'operations' | 'output' | 'state' | 'danger';
export type LayerContextSubmenuId = 'arrange' | 'boolean' | 'copy-to' | 'convert-to' | 'add-adjustment' | 'color-label';

export interface LayerContextActionState {
  canRunWorkflow: boolean;
  document: CanvasDocumentContractV3;
  hasEngine: boolean;
  hasSupportedContent: boolean;
  hasWorkflowBindings: boolean;
  interactionLocked: boolean;
  layer: CanvasLayerContract;
  /** The selected generation model's base; gates model-dependent actions like reference images. */
  modelBase: string | null;
  /** The panel's selection; an action on a selected layer applies to every selected node. */
  selectedIds: readonly string[];
  /** The model's answer for grouping `actionTargets`, asked by the menu that owns the engine. */
  canGroupSelection: boolean;
  /** The model's answer for removing `actionTargets`, asked the same way. */
  canDeleteSelection: boolean;
  /** Whether the whole multi-selection can merge into one raster, asked the same way. */
  canMergeSelection: boolean;
  /** A group above the layer hides it on the canvas; the layer's own flag cannot override that. */
  hiddenByAncestor: boolean;
}

export interface LayerContextActionEffects {
  reorder(kind: LayerStackMoveKind, actionId: LayerContextActionId): void;
  duplicate(): void;
  mergeSelected(): void;
  group(): void;
  openRename(): void;
  openRunWorkflow(): void;
  startSelectObject(layerId: string): void;
  startFilter(layerId: string): void;
  transform(): void;
  fitToBbox(): void;
  openProperties(): void;
  saveToAssets(): Promise<void>;
  copyToClipboard(): Promise<void>;
  cropToBbox(): Promise<void>;
  extractMaskedArea(): Promise<void>;
  booleanMerge(operation: BooleanRasterOperation): Promise<void>;
  copyTo(target: LayerType): void | Promise<void>;
  rasterize(): void;
  convertTo(target: LayerType): void;
  patchConfig(kind: LayerConfigPatchKind): void;
  addReferenceImage(): void;
  addMaskModifier(field: 'noise' | 'denoise'): void;
  addAdjustment(type: CanvasAdjustmentEntry['type']): void;
  addLayerRegion(): void;
  setColorLabel(label: CanvasColorLabel | null): void;
  mergeDown(): void;
  toggleVisibility(): void;
  toggleHidden(): void;
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
  /** CSS color for the item's icon (label swatches); the theme tone otherwise. */
  iconColor?: string;
  section: LayerContextMenuSectionId;
  submenu?: LayerContextSubmenuId;
  order: number;
  supportedLayerTypes: readonly LayerType[];
  tone?: 'danger';
  /** A raw hotkey string (e.g. `mod+]`) shown as a trailing hint on the item. */
  hint?: string;
  isVisible(context: LayerContextActionState): boolean;
  isEnabled(context: LayerContextActionState): boolean;
  handler(context: LayerContextActionRuntimeContext): void | Promise<void>;
  getDefaultLabel?(context: LayerContextActionState): string;
  getLabelKey?(context: LayerContextActionState): string;
  /** Interpolated into the label's plural key; the selection size for selection verbs. */
  getLabelCount?(context: LayerContextActionState): number;
}

export interface LayerContextAction {
  id: LayerContextActionId;
  labelKey: string;
  defaultLabel: string;
  labelCount?: number;
  icon: LucideIcon;
  iconColor?: string;
  section: LayerContextMenuSectionId;
  submenu?: LayerContextSubmenuId;
  order: number;
  tone?: 'danger';
  hint?: string;
  isDisabled: boolean;
  handler(context: LayerContextActionRuntimeContext): void | Promise<void>;
}

export type LayerConfigPatchKind = 'control-transparency-effect' | 'regional-auto-negative';

const ALL_LAYER_TYPES = ['raster', 'control', 'inpaint_mask', 'regional_guidance'] as const;
/** The one add-adjustment catalog; the leaf action registry and the group menu both render from it. */
export const ADJUSTMENT_ADD_ITEMS: readonly {
  readonly defaultLabel: string;
  readonly icon: LucideIcon;
  readonly labelKey: string;
  readonly type: CanvasAdjustmentEntry['type'];
}[] = [
  {
    defaultLabel: 'Brightness/Contrast',
    icon: SunMediumIcon,
    labelKey: 'widgets.layers.modifiers.brightnessContrast',
    type: 'brightness-contrast',
  },
  { defaultLabel: 'Exposure', icon: ApertureIcon, labelKey: 'widgets.layers.adjustments.exposure', type: 'exposure' },
  { defaultLabel: 'Levels', icon: SlidersVerticalIcon, labelKey: 'widgets.layers.adjustments.levels', type: 'levels' },
  { defaultLabel: 'Curves', icon: SplineIcon, labelKey: 'widgets.layers.adjustments.curves', type: 'curves' },
  { defaultLabel: 'Saturation', icon: DropletIcon, labelKey: 'widgets.layers.adjustments.saturation', type: 'hsl' },
  { defaultLabel: 'Hue', icon: RainbowIcon, labelKey: 'widgets.layers.adjustments.hue', type: 'hue' },
  { defaultLabel: 'Invert', icon: ContrastIcon, labelKey: 'widgets.layers.adjustments.invert', type: 'invert' },
];

const RASTER_ONLY = ['raster'] as const;
const CONTROL_ONLY = ['control'] as const;
const RASTER_AND_CONTROL = ['raster', 'control'] as const;
const INPAINT_ONLY = ['inpaint_mask'] as const;
const REGIONAL_ONLY = ['regional_guidance'] as const;

const alwaysVisible = (): boolean => true;
const isInteractionFree = (context: LayerContextActionState): boolean => !context.interactionLocked;
const layerEntry = (context: LayerContextActionState) => getDocumentIndex(context.document).byId.get(context.layer.id);
/** The nodes a selection-wide action applies to: the whole selection when the layer is in it. */
export const actionTargets = (context: Pick<LayerContextActionState, 'layer' | 'selectedIds'>): readonly string[] =>
  context.selectedIds.includes(context.layer.id) ? context.selectedIds : [context.layer.id];

const targetCount = (context: LayerContextActionState): number => actionTargets(context).length;
const isMultiTarget = (context: LayerContextActionState): boolean => targetCount(context) > 1;
const targetNodes = (context: LayerContextActionState) => {
  const index = getDocumentIndex(context.document);
  return actionTargets(context).flatMap((id) => index.byId.get(id)?.node ?? []);
};
const allTargetsEnabled = (context: LayerContextActionState): boolean =>
  targetNodes(context).every((node) => node.isEnabled);
const allTargetsLocked = (context: LayerContextActionState): boolean =>
  targetNodes(context).every((node) => node.isLocked);
/** Locked in its own right or by a group above it: content edits are refused either way. */
const isLayerFrozen = (context: LayerContextActionState): boolean =>
  context.layer.isLocked || (layerEntry(context)?.ancestorsLocked ?? false);
const isLayerMutable = (context: LayerContextActionState): boolean =>
  isInteractionFree(context) && !isLayerFrozen(context);
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
    (layer.source.type === 'shape' && !isEmptyPolygonShape(layer.source)));

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
    source.type === 'text' || source.type === 'gradient' || (source.type === 'shape' && !isEmptyPolygonShape(source))
  );
};

/** Where the layer sits among its siblings (index 0 = top), or null when absent. */
const siblingPosition = (context: LayerContextActionState): { index: number; count: number } | null => {
  const index = getDocumentIndex(context.document);
  const entry = index.byId.get(context.layer.id);
  if (!entry) {
    return null;
  }
  const parent = entry.parentId === null ? null : index.byId.get(entry.parentId)!.node;
  const count =
    parent && parent.type === 'group' ? parent.children.length : context.document.stacks[entry.stack].length;
  return { count, index: entry.siblingIndex };
};

const canMoveForward = (context: LayerContextActionState): boolean => {
  const position = siblingPosition(context);
  return isInteractionFree(context) && !!position && position.index > 0;
};

const canMoveBackward = (context: LayerContextActionState): boolean => {
  const position = siblingPosition(context);
  return isInteractionFree(context) && !!position && position.index < position.count - 1;
};

const hasMergeableLayerBelow = (context: LayerContextActionState): boolean =>
  canMergeLayerDown(context.document, context.layer.id, true);

const isBooleanRasterLayer = (layer: CanvasLayerContract | null): boolean =>
  !!layer && layer.isEnabled && layer.type === 'raster' && isPixelBackedLayer(layer);

const hasBooleanRasterPair = (context: LayerContextActionState): boolean =>
  isBooleanRasterLayer(context.layer) && isBooleanRasterLayer(lookupLayerBelow(context.document, context.layer.id));

export const LAYER_CONTEXT_ACTION_DEFINITIONS: readonly LayerContextActionDefinition[] = [
  {
    defaultLabel: 'Move to front',
    handler: ({ effects }) => effects.reorder('front', 'move-to-front'),
    icon: ArrowUpToLineIcon,
    hint: 'mod+shift+]',
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
    hint: 'mod+]',
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
    hint: 'mod+[',
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
    hint: 'mod+shift+[',
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
    getDefaultLabel: (context) => (isMultiTarget(context) ? `Duplicate ${targetCount(context)} layers` : 'Duplicate'),
    getLabelCount: targetCount,
    getLabelKey: (context) =>
      isMultiTarget(context) ? 'widgets.layers.actions.duplicateCount' : 'widgets.layers.actions.duplicate',
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
    defaultLabel: 'Group layers',
    getDefaultLabel: (context) => (isMultiTarget(context) ? `Group ${targetCount(context)} layers` : 'Group layers'),
    getLabelCount: targetCount,
    getLabelKey: (context) =>
      isMultiTarget(context) ? 'widgets.layers.actions.groupCount' : 'widgets.layers.actions.group',
    handler: ({ effects }) => effects.group(),
    icon: FolderPlusIcon,
    id: 'group',
    isEnabled: (context) => isInteractionFree(context) && context.canGroupSelection,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.group',
    order: 20,
    section: 'quick',
    supportedLayerTypes: ALL_LAYER_TYPES,
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
  ...ADJUSTMENT_ADD_ITEMS.map((item, index): LayerContextActionDefinition => ({
    defaultLabel: item.defaultLabel,
    handler: ({ effects }) => effects.addAdjustment(item.type),
    icon: item.icon,
    id: `add-${item.type}` as LayerContextActionId,
    isEnabled: isLayerMutable,
    isVisible: alwaysVisible,
    labelKey: item.labelKey,
    order: 31 + index * 0.05,
    section: 'primary',
    submenu: 'add-adjustment',
    supportedLayerTypes: RASTER_ONLY,
  })),
  {
    defaultLabel: 'Filter',
    handler: ({ effects, layer }) => effects.startFilter(layer.id),
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
    isVisible: hasFilterableLayerContent,
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
    defaultLabel: 'Add reference image',
    handler: ({ effects }) => effects.addReferenceImage(),
    icon: ImagePlusIcon,
    id: 'add-reference-image',
    isEnabled: isLayerMutable,
    isVisible: (context) => canAddRegionalReferenceImage(context.modelBase),
    labelKey: 'widgets.layers.regionalGuidance.addReferenceImage',
    order: 55,
    section: 'primary',
    supportedLayerTypes: REGIONAL_ONLY,
  },
  {
    defaultLabel: 'Add regenerate region',
    handler: ({ effects }) => effects.addLayerRegion(),
    icon: WandSparklesIcon,
    id: 'add-regenerate-region',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'raster' && context.layer.inpaint === undefined,
    labelKey: 'widgets.layers.actions.addRegenerateRegion',
    order: 32,
    section: 'primary',
    supportedLayerTypes: RASTER_ONLY,
  },
  {
    defaultLabel: 'Add noise',
    handler: ({ effects }) => effects.addMaskModifier('noise'),
    icon: WavesIcon,
    id: 'add-noise',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'inpaint_mask' && context.layer.noise === undefined,
    labelKey: 'widgets.layers.actions.addNoise',
    order: 55,
    section: 'primary',
    supportedLayerTypes: INPAINT_ONLY,
  },
  {
    defaultLabel: 'Add denoise limit',
    handler: ({ effects }) => effects.addMaskModifier('denoise'),
    icon: GaugeIcon,
    id: 'add-denoise-limit',
    isEnabled: isLayerMutable,
    isVisible: (context) => context.layer.type === 'inpaint_mask' && context.layer.denoise === undefined,
    labelKey: 'widgets.layers.actions.addDenoiseLimit',
    order: 56,
    section: 'primary',
    supportedLayerTypes: INPAINT_ONLY,
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
    defaultLabel: 'Merge selected layers',
    handler: ({ effects }) => effects.mergeSelected(),
    icon: MergeIcon,
    id: 'merge-selected',
    isEnabled: (context) => isInteractionFree(context) && context.canMergeSelection,
    isVisible: isMultiTarget,
    labelKey: 'widgets.layers.actions.mergeSelected',
    order: 1,
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
    isVisible: (context) => isPixelBackedLayer(context.layer),
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
    isVisible: (context) => isPixelBackedLayer(context.layer) || context.layer.type === 'regional_guidance',
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
    isVisible: (context) => isPixelBackedLayer(context.layer) || context.layer.type === 'inpaint_mask',
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
    isVisible: (context) => isPixelBackedLayer(context.layer),
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
    isVisible: (context) => isPixelBackedLayer(context.layer),
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
    defaultLabel: 'Save layer to uploads',
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
    defaultLabel: 'Enable/disable',
    getDefaultLabel: (context) =>
      isMultiTarget(context)
        ? allTargetsEnabled(context)
          ? 'Disable selected'
          : 'Enable selected'
        : context.layer.isEnabled
          ? 'Disable layer'
          : 'Enable layer',
    getIcon: (context) => (allTargetsEnabled(context) ? CircleOffIcon : CircleIcon),
    getLabelKey: (context) =>
      isMultiTarget(context)
        ? allTargetsEnabled(context)
          ? 'widgets.layers.actions.disableSelected'
          : 'widgets.layers.actions.enableSelected'
        : context.layer.isEnabled
          ? 'widgets.layers.actions.disableLayer'
          : 'widgets.layers.actions.enableLayer',
    handler: ({ effects }) => effects.toggleVisibility(),
    icon: CircleIcon,
    id: 'toggle-visibility',
    isEnabled: isInteractionFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleVisibility',
    order: 0,
    section: 'state',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Show/hide layer',
    getDefaultLabel: (context) => (isNodeHidden(context.layer) ? 'Show layer' : 'Hide layer'),
    getIcon: (context) => (isNodeHidden(context.layer) ? EyeIcon : EyeOffIcon),
    getLabelKey: (context) =>
      isNodeHidden(context.layer) ? 'widgets.layers.actions.showLayer' : 'widgets.layers.actions.hideLayer',
    handler: ({ effects }) => effects.toggleHidden(),
    icon: EyeOffIcon,
    id: 'toggle-hidden',
    isEnabled: (context) => isInteractionFree(context) && !context.hiddenByAncestor,
    isVisible: (context) => isHideableLayer(context.layer),
    labelKey: 'widgets.layers.actions.toggleHidden',
    order: 5,
    section: 'state',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Toggle lock',
    getDefaultLabel: (context) =>
      isMultiTarget(context)
        ? allTargetsLocked(context)
          ? 'Unlock selected'
          : 'Lock selected'
        : context.layer.isLocked
          ? 'Unlock'
          : 'Lock',
    getIcon: (context) => (allTargetsLocked(context) ? LockOpenIcon : LockIcon),
    getLabelKey: (context) =>
      isMultiTarget(context)
        ? allTargetsLocked(context)
          ? 'widgets.layers.actions.unlockSelected'
          : 'widgets.layers.actions.lockSelected'
        : context.layer.isLocked
          ? 'widgets.layers.actions.unlock'
          : 'widgets.layers.actions.lock',
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
  // Color labels are organizational, so a locked layer still takes one.
  ...COLOR_LABEL_ITEMS.map((item, index): LayerContextActionDefinition => ({
    defaultLabel: item.defaultLabel,
    handler: ({ effects }) => effects.setColorLabel(item.value),
    icon: CircleIcon,
    iconColor: item.hex,
    id: `color-label-${item.value}`,
    isEnabled: isInteractionFree,
    isVisible: alwaysVisible,
    labelKey: item.labelKey,
    order: 15 + index,
    section: 'state',
    submenu: 'color-label',
    supportedLayerTypes: ALL_LAYER_TYPES,
  })),
  {
    defaultLabel: 'None',
    handler: ({ effects }) => effects.setColorLabel(null),
    icon: CircleOffIcon,
    id: 'color-label-none',
    isEnabled: (context) => isInteractionFree(context) && context.layer.colorLabel !== undefined,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.labels.none',
    order: 15 + COLOR_LABEL_ITEMS.length,
    section: 'state',
    submenu: 'color-label',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Delete',
    getDefaultLabel: (context) => (isMultiTarget(context) ? `Delete ${targetCount(context)} layers` : 'Delete'),
    getLabelCount: targetCount,
    getLabelKey: (context) =>
      isMultiTarget(context) ? 'widgets.layers.actions.deleteCount' : 'widgets.layers.actions.delete',
    handler: ({ effects }) => effects.delete(),
    icon: Trash2Icon,
    id: 'delete',
    isEnabled: (context) => isInteractionFree(context) && context.canDeleteSelection,
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
    hint: definition.hint,
    icon: definition.getIcon?.(context) ?? definition.icon,
    iconColor: definition.iconColor,
    id: definition.id,
    isDisabled: !definition.isEnabled(context),
    labelCount: definition.getLabelCount?.(context),
    labelKey: definition.getLabelKey?.(context) ?? definition.labelKey,
    order: definition.order,
    section: definition.section,
    submenu: definition.submenu,
    tone: definition.tone,
  }));
