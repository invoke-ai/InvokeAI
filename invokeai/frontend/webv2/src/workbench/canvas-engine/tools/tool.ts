/** Navigation changes only the viewport; painting edits caches without move dispatch, then emits one commit event. */

import type {
  CanvasEditRefusal,
  ColorSamplePhase,
  StructuralCommitResult,
} from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3, CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeInsertionAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import type { LayerStackKind } from '@workbench/canvas-engine/document/layerStacks';
import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { CompositeOptions } from '@workbench/canvas-engine/render/compositor';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { InvalidatePayload } from '@workbench/canvas-engine/render/scheduler';
import type { SamInteractionState, SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { FloatingSelection, FloatLiftResult } from '@workbench/canvas-engine/selection/floatingSelection';
import type { SelectionCommit } from '@workbench/canvas-engine/selection/selectionState';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PlacedSurface, PointerInput, PointerModifiers, Rect, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

/** Before/after images sized to `dirtyRect`, for exact undo/redo and persistence. */
export interface StrokeCommittedEvent {
  layerId: string;
  /** Layer-local integer bounds, clamped to the paintable region. */
  dirtyRect: Rect;
  beforeImageData: ImageData;
  afterImageData: ImageData;
  /** `shape-cutout` is a shape whose footprint was erased. */
  tool: 'brush' | 'eraser' | 'shape' | 'shape-cutout';
  /** Lets stroke history remove and recreate both layer and pixels. */
  createdLayer?: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor };
}

export interface PixelEditPatch {
  rect: Rect;
  before: ImageData;
  after: ImageData;
}

/** A refused `grow` means the caller must restore its pixels and cancel. */
export interface StrokeEdit {
  grow(bytes: number): boolean;
  commit(event: StrokeCommittedEvent): boolean;
  cancel(): void;
}

/** A `false` commit leaves it open: restore the touched pixels, then cancel to undo the materialization. */
export interface PixelEditTransaction extends StrokeEdit {
  readonly layerId: string;
  commitPatch(label: string, patch: PixelEditPatch): boolean;
}

/** Render-time only; move sets position, transform the full values. */
export interface LayerTransformOverride {
  x: number;
  y: number;
  scaleX?: number;
  scaleY?: number;
  rotation?: number;
}

/** Layer-local, per layer id. */
export interface PaintLineEnds {
  get(layerId: string): Vec2 | undefined;
  set(layerId: string, point: Vec2): void;
  clear(): void;
}

export interface ToolContext {
  viewport: Viewport;
  getDocument(): CanvasDocumentContractV3 | null;
  /** Includes the document's primary layer. */
  getSelectedLayerIds?(): readonly string[];
  invalidate(payload: InvalidatePayload): void;
  /** Runs before the next composited frame so its invalidations render in it; returns a cancel. */
  scheduleFrame(task: () => void): () => void;
  dispatch(action: CanvasProjectMutation): void;
  captureInsertionAnchor(stack: LayerStackKind, aboveId: string | null): CanvasNodeInsertionAnchor;
  commitStructural(
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation
  ): StructuralCommitResult;
  /** Clear on commit or cancel. */
  setLayerTransformOverride(layerId: string, override: LayerTransformOverride | null): void;
  beginTransformSession?(layerId: string): void;
  beginPixelEdit?(layerId: string): PixelEditTransaction | null;
  requestLayerRasterization?(layerId: string): void;
  updateTransformSession?(transform: LayerTransform): void;
  applyTransform?(): void;
  cancelTransform?(): void;
  /** No layer exists until one add commit. */
  openTextCreate?(docPoint: Vec2): void;
  /** Captures the existing text source as the exact undo baseline. */
  openTextEdit?(layerId: string): void;
  cancelTextEdit?(): void;
  backend: RasterBackend;
  layers: LayerCacheStore;
  /** Node-safe seam. */
  createPath2D: CreatePath2D;
  createLayerId(): string;
  stores: EngineStores;
  getSamInteraction?(): SamInteractionState | null;
  setOverlayCursor(cursor: OverlayCursor | null): void;
  showColorLoupe?(shown: boolean): void;
  /** `commit` fires on pointer up only, so a structural commit in the continuation is not `gesture-active`. */
  resolveColorSample?(hex: string, phase: ColorSamplePhase): boolean;
  /** Absent samples raw cached pixels. */
  sampleProviders?: Pick<CompositeOptions, 'adjustedSurface' | 'derivedSurfaces' | 'groupSurface'>;
  /** Layer-local, so moving the layer carries it; the engine forgets entries on replay, replacement and removal. */
  paintLineEnds?: PaintLineEnds;
  /** Ordinary pointermove does not reevaluate it. */
  updateCursor(): void;
  /** Null when refused; the engine reports why. */
  beginStrokeEdit(initialBytes?: number): StrokeEdit | null;
  reportEditRefusal?(refusal: CanvasEditRefusal): void;
  /** Does not mark the cache stale. */
  notifyLayerPainted(layerId: string): void;
  commitSelection?(commit: SelectionCommit): void;
  /** Captured once at pointerdown; null skips masking work. */
  getSelectionMask?(): PlacedSurface | null;
  /** Null means unclipped. */
  getStrokeClipRect?(): Rect | null;
  updateSamInput?(input: SamVisualInput): void;
  /** A refused lift has already been reported. */
  liftFloatingSelection?(layerId: string): FloatLiftResult;
  getFloatingSelection?(): FloatingSelection | null;
  /** Layer-local; a transform its undo budget cannot cover keeps the last. */
  setFloatingTransform?(transform: LayerTransform): void;
  commitFloatingSelection?(): void;
  cancelFloatingSelection?(): void;
  /** So a float drags correctly under a rotated or scaled layer. */
  documentDeltaToLayerLocal?(layerId: string, delta: Vec2): Vec2;
  isPointInSelection?(point: Vec2): boolean;
  /** For tools that only know after {@link Tool.editsSelectedLayer} was asked; a no-op outside a gesture. */
  beginSelectedLayerEdit?(): void;
}

export interface ToolActivationOptions {
  temporary?: boolean;
}

export interface Tool {
  readonly id: ToolId;
  /** A multi-press creation session may remain active between pointer gestures. */
  hasSession?(): boolean;
  /** Called when the tool becomes active. */
  onActivate?(ctx: ToolContext, opts?: ToolActivationOptions): void;
  onDeactivate?(ctx: ToolContext, opts?: ToolActivationOptions): void;
  onPointerDown?(ctx: ToolContext, input: PointerInput): void;
  /** `batch` holds at least one sample, the last equal to `input`. */
  onPointerMove?(ctx: ToolContext, input: PointerInput, batch: readonly PointerInput[]): void;
  onPointerUp?(ctx: ToolContext, input: PointerInput): void;
  onPointerCancel?(ctx: ToolContext): void;
  /** A handled `cancel` ends the Escape ladder, so Escape never also clears the selection. */
  onKeyCommand?(ctx: ToolContext, command: 'apply' | 'cancel'): boolean;
  /** Owned modifier changes; returning true keeps a modifier from activating a temporary tool. */
  onModifierKey?(ctx: ToolContext, code: 'Space' | 'Alt' | 'Shift', pressed: boolean): boolean;
  /** Drop or reconcile an unfinished gesture when its document target changes. Never dispatch from this hook. */
  onDocumentChanged?(ctx: ToolContext): void;

  /** Wheel over the canvas; `screenAnchor` is the CSS-pixel cursor position. */
  onWheel?(ctx: ToolContext, deltaY: number, screenAnchor: { x: number; y: number }, modifiers: PointerModifiers): void;
  cursor?(ctx: ToolContext): string;
  /** The pointer pipeline must not turn this tool's alt-hold into the temporary color picker. */
  readonly usesAltKey?: boolean;
  /** Asked once as each gesture starts; absent means unknown then. Drives overlay stacks above staged previews. */
  editsSelectedLayer?(ctx: ToolContext): boolean;
}
