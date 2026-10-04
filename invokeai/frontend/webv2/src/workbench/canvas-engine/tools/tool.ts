/**
 * Tools handle normalized input through engine context and request rendering. Navigation changes only viewport.
 * Painting may create a layer at gesture start, edits caches without move dispatch, and emits one {@link
 * StrokeCommittedEvent} for downstream history/persistence.
 */

import type { StructuralCommitResult } from '@workbench/canvas-engine/capabilities';
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

/** One completed-stroke event carries before/after images sized to dirtyRect for exact undo/redo and persistence. */
export interface StrokeCommittedEvent {
  /** The layer that received the stroke. */
  layerId: string;
  /** The painted region in LAYER-LOCAL space (integer bounds, clamped to the paintable region). */
  dirtyRect: Rect;
  /** Cache pixels within `dirtyRect` before the stroke. */
  beforeImageData: ImageData;
  /** Cache pixels within `dirtyRect` after the stroke. */
  afterImageData: ImageData;
  /** Which tool produced the stroke; a `shape` is one drawn as pixels onto a paint layer. */
  tool: 'brush' | 'eraser' | 'shape';
  /** Optional auto-created layer and insertion placement let stroke history remove/recreate both layer and pixels. */
  createdLayer?: { layer: CanvasLayerContract; anchor: CanvasNodeInsertionAnchor };
}

export interface PixelEditPatch {
  rect: Rect;
  before: ImageData;
  after: ImageData;
}

/**
 * An admitted live pixel edit. Its undo footprint is admitted before pixels change: `grow` admits more before the
 * edit expands, and a refusal means the caller must restore its pixels and cancel. `commit` records the
 * already-applied pixels as one undo step and reports whether it did; `cancel` ends it unrecorded.
 */
export interface StrokeEdit {
  grow(bytes: number): boolean;
  commit(event: StrokeCommittedEvent): boolean;
  cancel(): void;
}

/**
 * A pixel edit that may first materialize its layer (control layers, raster images) in place. A `false` commit
 * leaves it open: the caller restores the pixels it touched, then cancels, which undoes the materialization.
 */
export interface PixelEditTransaction extends StrokeEdit {
  readonly layerId: string;
  commitPatch(label: string, patch: PixelEditPatch): boolean;
}

/**
 * Transient render-time transforms leave the mirror unchanged. Move sets position only; transform supplies full
 * values.
 */
export interface LayerTransformOverride {
  x: number;
  y: number;
  scaleX?: number;
  scaleY?: number;
  rotation?: number;
}

/** Everything a tool is allowed to reach, injected by the engine. */
export interface ToolContext {
  /** The pan/zoom viewport. */
  viewport: Viewport;
  /** The current mirrored document, or `null` when none is available. */
  getDocument(): CanvasDocumentContractV3 | null;
  /** Selected layer ids from the Layers panel, including the document's primary layer. */
  getSelectedLayerIds?(): readonly string[];
  /** Requests a re-render for the given flags. */
  invalidate(payload: InvalidatePayload): void;
  /** Runs `task` once before the next composited frame so its invalidations render in that frame; returns a cancel. */
  scheduleFrame(task: () => void): () => void;
  /** Reducer bridge. Painting tools use it for the single gesture-start `addCanvasLayer`. */
  dispatch(action: CanvasProjectMutation): void;
  /** Where a layer the tool creates lands: above `aboveId` when it belongs to `stack`, else the stack top. */
  captureInsertionAnchor(stack: LayerStackKind, aboveId: string | null): CanvasNodeInsertionAnchor;
  /** Applies forward structural action now and records inverse/forward replay in canvas history. */
  commitStructural(
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation
  ): StructuralCommitResult;
  /** Set or clear transient compositor/overlay transforms without document mutation; clear on commit/cancel. */
  setLayerTransformOverride(layerId: string, override: LayerTransformOverride | null): void;
  beginTransformSession?(layerId: string): void;
  /** Prepares direct or materializing pixel editing for a selected control or raster-image layer. */
  beginPixelEdit?(layerId: string): PixelEditTransaction | null;
  /** Starts reserved source materialization when an offscreen durable paint target is not ready yet. */
  requestLayerRasterization?(layerId: string): void;
  /** Updates the active transform session's live transform (drag or numeric edit). */
  updateTransformSession?(transform: LayerTransform): void;
  /** Commits image parameters or paint bake as one undo entry, then clears the session. */
  applyTransform?(): void;
  /** Cancels the active transform session (drops the preview, no dispatch). */
  cancelTransform?(): void;
  /** Optional creation session at document point with tool defaults; no layer exists until one add commit. */
  openTextCreate?(docPoint: Vec2): void;
  /** Optional edit session captures the existing text source as exact undo baseline. */
  openTextEdit?(layerId: string): void;
  /** Cancels the active text-editing session (drops it, no dispatch). */
  cancelTextEdit?(): void;
  /** The raster backend, for allocating scratch stroke surfaces. */
  backend: RasterBackend;
  /** The per-layer raster cache; painting tools fill directly into a layer's surface. */
  layers: LayerCacheStore;
  /** Builds a `Path2D` (node-safe seam; the engine passes `(d) => new Path2D(d)`). */
  createPath2D: CreatePath2D;
  /** Mints a fresh layer id for an auto-created paint layer. */
  createLayerId(): string;
  /** The transient engine stores (tool options live here). */
  stores: EngineStores;
  /** Reads core visual Select Object interaction state without depending on application sessions. */
  getSamInteraction?(): SamInteractionState | null;
  /** Sets (or clears) the brush cursor ring drawn on the overlay. */
  setOverlayCursor(cursor: OverlayCursor | null): void;
  /** Shows or hides the color picker's loupe, which follows the pointer while shown; optional in test harnesses. */
  showColorLoupe?(shown: boolean): void;
  /**
   * Route samples to one-shot claim, then persistent workbench target. Return whether consumed; otherwise picker
   * falls back to brush color. Optional in test harnesses.
   */
  resolveColorSample?(hex: string): boolean;
  /** Compositor providers that make sampling WYSIWYG; absent ⇒ raw cached pixels. */
  sampleProviders?: Pick<CompositeOptions, 'adjustedSurface' | 'derivedSurfaces' | 'groupSurface'>;
  /**
   * Settles a pending one-shot sample with the gesture's stashed color. Called
   * on pointer up only, so a structural commit in the requester's continuation
   * is not refused as `gesture-active`.
   */
  commitColorSample?(): void;
  /** Drops the stashed sample (gesture cancel, fresh press) without settling the request. */
  discardColorSample?(): void;
  /**
   * Refresh the input element cursor when tool state changes; ordinary pointermove does not automatically
   * reevaluate it.
   */
  updateCursor(): void;
  /** Admits a live pixel edit on a paint layer; null when refused (the engine reports why). */
  beginStrokeEdit(initialBytes?: number): StrokeEdit | null;
  /** Bumps a layer's cache version (without marking it stale) after a direct paint, and recomposites. */
  notifyLayerPainted(layerId: string): void;
  commitSelection?(commit: SelectionCommit): void;
  /**
   * Optional placed document-space selection mask; paint tools capture it once at pointerdown. Null avoids masking
   * work.
   */
  getSelectionMask?(): PlacedSurface | null;
  /** Optional document-space stroke clip captured once at pointerdown; null means unclipped. */
  getStrokeClipRect?(): Rect | null;
  /** Updates visual SAM input for the active engine-owned Select Object session. */
  updateSamInput?(input: SamVisualInput): void;
  /** Optional lift of selected layer pixels into a float; a refused lift has already been reported. */
  liftFloatingSelection?(layerId: string): FloatLiftResult;
  /** The live floating selection, or `null`. */
  getFloatingSelection?(): FloatingSelection | null;
  /** Sets the float's live transform (LAYER-LOCAL space); one its undo budget cannot cover keeps the last. */
  setFloatingTransform?(transform: LayerTransform): void;
  /** Bakes the float back into its layer as one undoable entry. */
  commitFloatingSelection?(): void;
  /** Puts the float's pixels back untouched, pushing no history. */
  cancelFloatingSelection?(): void;
  /**
   * Converts a DOCUMENT-space delta into `layerId`'s LOCAL space, so a pointer
   * drag moves a float correctly under a rotated/scaled layer.
   */
  documentDeltaToLayerLocal?(layerId: string, delta: Vec2): Vec2;
  /** True if `point` (document space) falls inside the live selection. */
  isPointInSelection?(point: Vec2): boolean;
}

/** Temporary modifier switches/restores preserve sessions; real switches or disposal tear them down. */
export interface ToolActivationOptions {
  /** True for a pipeline modifier-hold switch (and its matching restore); absent/false for a real switch. */
  temporary?: boolean;
}

/** A stateless-to-the-engine interaction handler. Implementations may hold private drag state. */
export interface Tool {
  readonly id: ToolId;
  /** Called when the tool becomes active. */
  onActivate?(ctx: ToolContext, opts?: ToolActivationOptions): void;
  /** Called when the tool is deactivated (also on engine dispose). */
  onDeactivate?(ctx: ToolContext, opts?: ToolActivationOptions): void;
  onPointerDown?(ctx: ToolContext, input: PointerInput): void;
  /**
   * A pointer move. `batch` carries the coalesced samples for this move event
   * (always at least one; its last element equals `input`); tools that paint
   * consume the whole batch, navigation tools use only `input`.
   */
  onPointerMove?(ctx: ToolContext, input: PointerInput, batch: readonly PointerInput[]): void;
  onPointerUp?(ctx: ToolContext, input: PointerInput): void;
  /** The active gesture was cancelled (Esc, pointercancel, focus loss). */
  onPointerCancel?(ctx: ToolContext): void;
  /**
   * Optional session commands from the pipeline; tools may apply/cancel while gesture cancellation remains
   * separate.
   */
  onKeyCommand?(ctx: ToolContext, command: 'apply' | 'cancel'): void;
  /** Wheel over the canvas; `screenAnchor` is the CSS-pixel cursor position. */
  onWheel?(ctx: ToolContext, deltaY: number, screenAnchor: { x: number; y: number }, modifiers: PointerModifiers): void;
  /** The CSS cursor to show while this tool is active. */
  cursor?(ctx: ToolContext): string;
  /**
   * Alt is one of this tool's gesture modifiers (selection ops), so the pointer
   * pipeline must not turn an alt-hold into the temporary color picker.
   */
  readonly usesAltKey?: boolean;
}
