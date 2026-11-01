import type {
  BooleanRasterOperation,
  CanvasAdjustmentEntry,
  CanvasColorLabel,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  InpaintMaskMergeEligibility,
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
  BookmarkIcon,
  BookmarkMinusIcon,
  CircleIcon,
  CircleOffIcon,
  CopyIcon,
  CropIcon,
  DropletIcon,
  EyeClosedIcon,
  EyeIcon,
  FilmIcon,
  FingerprintPatternIcon,
  FocusIcon,
  EyeOffIcon,
  FolderPlusIcon,
  GaugeIcon,
  HazeIcon,
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
  UngroupIcon,
  WavesIcon,
  SlidersHorizontalIcon,
  Trash2Icon,
  WorkflowIcon,
} from 'lucide-react';

import { COLOR_LABEL_ITEMS } from './colorLabels';
import { removeFromGroupCommand } from './layerGroupCommands';
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
  | 'add-texture'
  | 'add-clarity'
  | 'add-dehaze'
  | 'add-noise-adjustment'
  | 'merge-selected'
  | 'merge-masks'
  | 'move-to-front'
  | 'move-forward'
  | 'move-backward'
  | 'move-to-back'
  | 'duplicate'
  | 'group'
  | 'remove-from-group'
  | 'rename'
  | 'transform'
  | 'fit-to-bbox'
  | 'save-to-board'
  | 'copy-to-clipboard'
  | 'crop-to-bbox'
  | 'extract-masked-area'
  | 'filter'
  | 'select-object'
  | 'run-workflow'
  | 'use-as-control-layer'
  | 'intersect'
  | 'cutout'
  | 'cutaway'
  | 'exclude'
  | 'copy-to-raster'
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
  | 'bookmark-quick-switch'
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
  operationLocked: boolean;
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
  /** Whether merge masks would take `actionTargets`, and if not, whether only their generation settings differ. */
  mergeMasksEligibility: InpaintMaskMergeEligibility;
  /** The model's answer for lifting `actionTargets` out of their group. */
  canRemoveFromGroup: boolean;
  /** A group above the layer hides it on the canvas; the layer's own flag cannot override that. */
  hiddenByAncestor: boolean;
  /** Whether Q jumps to this layer: the engine's transient quick-switch bookmark names it. */
  isQuickSwitchBookmark: boolean;
}

export interface LayerContextActionEffects {
  reorder(kind: LayerStackMoveKind, actionId: LayerContextActionId): void;
  duplicate(): void;
  mergeSelected(): void;
  mergeMasks(): Promise<void>;
  group(): void;
  removeFromGroup(): void;
  openRename(): void;
  openRunWorkflow(): void;
  startSelectObject(layerId: string): void;
  startFilter(layerId: string): void;
  transform(): void;
  fitToBbox(): void;
  openProperties(): void;
  saveToBoard(board: LayerContextBoardChoice): Promise<void>;
  copyToClipboard(): Promise<void>;
  cropToBbox(): Promise<void>;
  extractMaskedArea(): Promise<void>;
  booleanMerge(operation: BooleanRasterOperation): Promise<void>;
  copyTo(target: LayerType): void | Promise<void>;
  rasterize(): Promise<void>;
  convertTo(target: LayerType): void | Promise<void>;
  patchConfig(kind: LayerConfigPatchKind): void;
  addReferenceImage(): void;
  addMaskModifier(field: 'noise' | 'denoise'): void;
  addAdjustment(type: CanvasAdjustmentEntry['type']): void;
  addLayerRegion(): void;
  setColorLabel(label: CanvasColorLabel | null): void;
  mergeDown(): Promise<void>;
  toggleVisibility(): void;
  toggleHidden(): void;
  toggleLock(): void;
  toggleQuickSwitchBookmark(): void;
  delete(): void;
}

/** A gallery board picked in a `picker: 'board'` action's submenu; `id` is `'none'` for Uncategorized. */
export interface LayerContextBoardChoice {
  id: string;
  label: string;
}

export interface LayerContextActionRuntimeContext extends LayerContextActionState {
  effects: LayerContextActionEffects;
  /** The board a `picker: 'board'` action runs with; absent for every other action. */
  board?: LayerContextBoardChoice;
}

interface LayerContextActionDefinitionBase {
  id: LayerContextActionId;
  labelKey: string;
  defaultLabel: string;
  icon: LucideIcon;
  getIcon?(context: LayerContextActionState): LucideIcon;
  /** CSS color for the item's icon (label swatches); the theme tone otherwise. */
  iconColor?: string;
  section: LayerContextMenuSectionId;
  order: number;
  supportedLayerTypes: readonly LayerType[];
  tone?: 'danger';
  /** A raw hotkey string (e.g. `mod+]`) shown as a trailing hint on the item. */
  hint?: string;
  /** A second line under the label explaining what the action does, shown whether or not it is enabled. */
  descriptionKey?: string;
  /** Why the disabled action cannot run here; asked only when `isEnabled` is false. */
  getUnavailableReasonKey?(context: LayerContextActionState): string | undefined;
  isVisible(context: LayerContextActionState): boolean;
  isEnabled(context: LayerContextActionState): boolean;
  handler(context: LayerContextActionRuntimeContext): void | Promise<void>;
  getDefaultLabel?(context: LayerContextActionState): string;
  getLabelKey?(context: LayerContextActionState): string;
  /** Interpolated into the label's plural key; the selection size for selection verbs. */
  getLabelCount?(context: LayerContextActionState): number;
}

/**
 * A picker action is its own submenu, so it cannot also sit in a static one: there it would render as a plain item
 * that runs without a board.
 */
export type LayerContextActionDefinition = LayerContextActionDefinitionBase &
  (
    | { picker?: never; submenu?: LayerContextSubmenuId }
    /** Renders the action as a submenu of the account's gallery boards; the chosen board runs it. */
    | { picker: 'board'; submenu?: never }
  );

export interface LayerContextAction {
  id: LayerContextActionId;
  labelKey: string;
  defaultLabel: string;
  labelCount?: number;
  icon: LucideIcon;
  iconColor?: string;
  section: LayerContextMenuSectionId;
  submenu?: LayerContextSubmenuId;
  picker?: 'board';
  order: number;
  tone?: 'danger';
  hint?: string;
  descriptionKey?: string;
  isDisabled: boolean;
  /** Set only on a disabled action whose reason is known; the menu keeps such an item reachable and says why. */
  unavailableReasonKey?: string;
  handler(context: LayerContextActionRuntimeContext): void | Promise<void>;
}

export type LayerConfigPatchKind = 'control-transparency-effect' | 'regional-auto-negative';

const ALL_LAYER_TYPES = ['raster', 'control', 'inpaint_mask', 'regional_guidance'] as const;
type AdjustmentAddItem = {
  readonly defaultLabel: string;
  readonly icon: LucideIcon;
  readonly labelKey: string;
  readonly type: CanvasAdjustmentEntry['type'];
};

/** The one add-adjustment catalog; the leaf action registry and the group menu both render from it. */
export const ADJUSTMENT_ADD_ITEMS: readonly AdjustmentAddItem[] = [
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
  // Detail adjustments, finest scale first.
  {
    defaultLabel: 'Texture',
    icon: FingerprintPatternIcon,
    labelKey: 'widgets.layers.adjustments.texture',
    type: 'texture',
  },
  { defaultLabel: 'Clarity', icon: FocusIcon, labelKey: 'widgets.layers.adjustments.clarity', type: 'clarity' },
  { defaultLabel: 'Dehaze', icon: HazeIcon, labelKey: 'widgets.layers.adjustments.dehaze', type: 'dehaze' },
  { defaultLabel: 'Noise', icon: FilmIcon, labelKey: 'widgets.layers.adjustments.noise', type: 'noise' },
];

/** An add-adjustment item's action id; noise is suffixed because `add-noise` is the inpaint mask's modifier. */
export const adjustmentAddActionId = (item: AdjustmentAddItem): LayerContextActionId =>
  item.type === 'noise' ? 'add-noise-adjustment' : (`add-${item.type}` as LayerContextActionId);

const RASTER_ONLY = ['raster'] as const;
const CONTROL_ONLY = ['control'] as const;
const RASTER_AND_CONTROL = ['raster', 'control'] as const;
const INPAINT_ONLY = ['inpaint_mask'] as const;
const REGIONAL_ONLY = ['regional_guidance'] as const;

const alwaysVisible = (): boolean => true;
const isOperationFree = (context: LayerContextActionState): boolean => !context.operationLocked;
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
const allTargetsInpaintMasks = (context: LayerContextActionState): boolean =>
  targetNodes(context).every((node) => node.type === 'inpaint_mask');
/** Locked in its own right or by a group above it: content edits are refused either way. */
const isLayerFrozen = (context: LayerContextActionState): boolean =>
  context.layer.isLocked || (layerEntry(context)?.ancestorsLocked ?? false);
const isLayerMutable = (context: LayerContextActionState): boolean =>
  isOperationFree(context) && !isLayerFrozen(context);
const CANVAS_NOT_READY_REASON = 'widgets.layers.actions.canvasNotReady';
const OPERATION_LOCKED_REASON = 'widgets.layers.actions.operationLocked';
/** Any target locked, by itself or by a group above it: the model refuses to move it out. */
const isAnyTargetFrozen = (context: LayerContextActionState): boolean => {
  const index = getDocumentIndex(context.document);
  return actionTargets(context).some((id) => {
    const entry = index.byId.get(id);
    return !!entry && (entry.node.isLocked || entry.ancestorsLocked);
  });
};
const hasReadablePixels = (context: LayerContextActionState): boolean =>
  context.hasEngine && context.hasSupportedContent && !context.operationLocked;
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
  return isOperationFree(context) && !!position && position.index > 0;
};

const canMoveBackward = (context: LayerContextActionState): boolean => {
  const position = siblingPosition(context);
  return isOperationFree(context) && !!position && position.index < position.count - 1;
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
    isEnabled: isOperationFree,
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
    isEnabled: (context) => isOperationFree(context) && context.canGroupSelection,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.group',
    order: 20,
    section: 'quick',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Remove from group',
    getUnavailableReasonKey: (context) =>
      !context.hasEngine
        ? CANVAS_NOT_READY_REASON
        : isAnyTargetFrozen(context)
          ? 'widgets.layers.actions.removeFromGroupLocked'
          : context.operationLocked
            ? OPERATION_LOCKED_REASON
            : undefined,
    handler: ({ effects }) => effects.removeFromGroup(),
    icon: UngroupIcon,
    id: 'remove-from-group',
    isEnabled: (context) => isOperationFree(context) && context.canRemoveFromGroup,
    isVisible: (context) => removeFromGroupCommand(context.document, actionTargets(context)) !== null,
    labelKey: 'widgets.layers.actions.removeFromGroup',
    order: 21,
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
    isEnabled: isOperationFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.rename',
    order: 20,
    section: 'primary',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Fit to frame',
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
    id: adjustmentAddActionId(item),
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
    // A control layer's map comes from its Detection row; the Filter operation refuses control layers.
    supportedLayerTypes: RASTER_ONLY,
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
    // Copies the pixels into a new control layer, whose Properties then ask what it should control.
    defaultLabel: 'Use as control layer',
    handler: ({ effects }) => effects.copyTo('control'),
    icon: SlidersHorizontalIcon,
    id: 'use-as-control-layer',
    isEnabled: hasReadablePixels,
    isVisible: (context) => isPixelBackedLayer(context.layer),
    labelKey: 'widgets.layers.actions.useAsControlLayer',
    order: 43,
    section: 'primary',
    supportedLayerTypes: RASTER_ONLY,
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
    descriptionKey: 'widgets.layers.actions.addRegenerateRegionDescription',
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
    isEnabled: (context) => isOperationFree(context) && context.canMergeSelection,
    // A mask-only selection merges through Merge masks instead; raster merge never accepts masks.
    isVisible: (context) => isMultiTarget(context) && !allTargetsInpaintMasks(context),
    labelKey: 'widgets.layers.actions.mergeSelected',
    order: 1,
    section: 'operations',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  {
    defaultLabel: 'Merge masks',
    getUnavailableReasonKey: (context) =>
      !context.hasEngine
        ? CANVAS_NOT_READY_REASON
        : context.mergeMasksEligibility === 'settings-differ'
          ? 'widgets.layers.actions.mergeMasksSettingsDiffer'
          : context.mergeMasksEligibility === 'ineligible'
            ? 'widgets.layers.actions.mergeMasksIneligible'
            : OPERATION_LOCKED_REASON,
    handler: ({ effects }) => effects.mergeMasks(),
    icon: MergeIcon,
    id: 'merge-masks',
    isEnabled: (context) => isOperationFree(context) && context.mergeMasksEligibility === 'eligible',
    isVisible: (context) => isMultiTarget(context) && allTargetsInpaintMasks(context),
    labelKey: 'widgets.layers.actions.mergeMasks',
    order: 1,
    section: 'operations',
    supportedLayerTypes: INPAINT_ONLY,
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
    defaultLabel: 'Crop layer to frame',
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
    defaultLabel: 'Save layer to board',
    handler: ({ board, effects }) => {
      if (!board) {
        throw new Error('Save layer to board runs with a chosen board');
      }
      return effects.saveToBoard(board);
    },
    icon: SaveIcon,
    id: 'save-to-board',
    isEnabled: hasReadablePixels,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.saveLayerToBoard',
    order: 10,
    picker: 'board',
    section: 'output',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  // One visibility model: hiding a layer takes it off the canvas and out of generation alike.
  {
    defaultLabel: 'Show or hide layer',
    getDefaultLabel: (context) =>
      isMultiTarget(context)
        ? allTargetsEnabled(context)
          ? 'Hide selected'
          : 'Show selected'
        : context.layer.isEnabled
          ? 'Hide layer'
          : 'Show layer',
    getIcon: (context) => (allTargetsEnabled(context) ? EyeOffIcon : EyeIcon),
    getLabelKey: (context) =>
      isMultiTarget(context)
        ? allTargetsEnabled(context)
          ? 'widgets.layers.actions.hideSelected'
          : 'widgets.layers.actions.showSelected'
        : context.layer.isEnabled
          ? 'widgets.layers.actions.hideLayer'
          : 'widgets.layers.actions.showLayer',
    handler: ({ effects }) => effects.toggleVisibility(),
    icon: EyeOffIcon,
    id: 'toggle-visibility',
    isEnabled: isOperationFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleActive',
    order: 0,
    section: 'state',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  // Overlays only: off the canvas to see the image under them, still used by generation.
  {
    defaultLabel: 'Hide on canvas only',
    getDefaultLabel: (context) => (isNodeHidden(context.layer) ? 'Show on canvas' : 'Hide on canvas only'),
    getIcon: (context) => (isNodeHidden(context.layer) ? EyeIcon : EyeClosedIcon),
    getLabelKey: (context) =>
      isNodeHidden(context.layer) ? 'widgets.layers.actions.showOnCanvas' : 'widgets.layers.actions.hideOnCanvas',
    handler: ({ effects }) => effects.toggleHidden(),
    icon: EyeClosedIcon,
    id: 'toggle-hidden',
    isEnabled: (context) => isOperationFree(context) && !context.hiddenByAncestor,
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
    isEnabled: isOperationFree,
    isVisible: alwaysVisible,
    labelKey: 'widgets.layers.actions.toggleLock',
    order: 10,
    section: 'state',
    supportedLayerTypes: ALL_LAYER_TYPES,
  },
  // The bookmark is a transient engine preference, so a locked layer still takes one.
  {
    defaultLabel: 'Bookmark for quick switch',
    getDefaultLabel: (context) =>
      context.isQuickSwitchBookmark ? 'Remove quick switch bookmark' : 'Bookmark for quick switch',
    getIcon: (context) => (context.isQuickSwitchBookmark ? BookmarkMinusIcon : BookmarkIcon),
    getLabelKey: (context) =>
      context.isQuickSwitchBookmark
        ? 'widgets.layers.actions.removeQuickSwitchBookmark'
        : 'widgets.layers.actions.bookmarkForQuickSwitch',
    handler: ({ effects }) => effects.toggleQuickSwitchBookmark(),
    hint: 'q',
    icon: BookmarkIcon,
    id: 'bookmark-quick-switch',
    isEnabled: isOperationFree,
    isVisible: (context) => context.hasEngine,
    labelKey: 'widgets.layers.actions.bookmarkForQuickSwitch',
    order: 12,
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
    isEnabled: isOperationFree,
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
    isEnabled: (context) => isOperationFree(context) && context.layer.colorLabel !== undefined,
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
    isEnabled: (context) => isOperationFree(context) && context.canDeleteSelection,
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
  ).map((definition) => {
    const isDisabled = !definition.isEnabled(context);
    return {
      defaultLabel: definition.getDefaultLabel?.(context) ?? definition.defaultLabel,
      descriptionKey: definition.descriptionKey,
      handler: definition.handler,
      hint: definition.hint,
      icon: definition.getIcon?.(context) ?? definition.icon,
      iconColor: definition.iconColor,
      id: definition.id,
      isDisabled,
      labelCount: definition.getLabelCount?.(context),
      labelKey: definition.getLabelKey?.(context) ?? definition.labelKey,
      order: definition.order,
      picker: definition.picker,
      section: definition.section,
      submenu: definition.submenu,
      tone: definition.tone,
      unavailableReasonKey: isDisabled ? definition.getUnavailableReasonKey?.(context) : undefined,
    };
  });
