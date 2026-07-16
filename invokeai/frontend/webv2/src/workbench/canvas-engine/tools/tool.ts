/**
 * The `Tool` seam: the engine routes normalized pointer/wheel input to whichever
 * tool is active. Tools are pure interaction handlers — they read the viewport
 * and mirrored document through the engine-provided {@link ToolContext} and
 * request re-renders via `invalidate`.
 *
 * Navigation tools (view) never dispatch and never touch pixels. Painting tools
 * (brush/eraser) reach the layer-cache surfaces and raster backend through the
 * same context, dispatch at most once per gesture (auto-creating a paint layer
 * on pointer-down when needed), and emit exactly one {@link StrokeCommittedEvent}
 * on commit — persistence/history are wired to that event downstream, not here.
 *
 * Zero React, zero import-time side effects.
 */

import type { EngineStores } from '@workbench/canvas-engine/engineStores';
import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { LayerCacheStore } from '@workbench/canvas-engine/render/layerCache';
import type { OverlayCursor } from '@workbench/canvas-engine/render/overlayRenderer';
import type { RasterBackend } from '@workbench/canvas-engine/render/raster';
import type { InvalidatePayload } from '@workbench/canvas-engine/render/scheduler';
import type { SamInteractionState, SamVisualInput } from '@workbench/canvas-engine/samInteraction';
import type { SelectionCommit } from '@workbench/canvas-engine/selection/selectionState';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';
import type { PlacedSurface, PointerInput, PointerModifiers, Rect, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

/**
 * Emitted once per completed brush/eraser gesture. Persistence (Task P2.2) and
 * history (Task P2.3) subscribe via `engine.tools.onStrokeCommitted`. `beforeImageData`
 * and `afterImageData` are both sized to `dirtyRect`, so an undo can restore the
 * pre-stroke pixels and a redo can re-apply the post-stroke pixels cheaply.
 */
export interface StrokeCommittedEvent {
  /** The layer that received the stroke. */
  layerId: string;
  /** The painted region in document space (integer bounds, clamped to the document). */
  dirtyRect: Rect;
  /** Cache pixels within `dirtyRect` before the stroke. */
  beforeImageData: ImageData;
  /** Cache pixels within `dirtyRect` after the stroke. */
  afterImageData: ImageData;
  /** Which tool produced the stroke. */
  tool: 'brush' | 'eraser';
  /**
   * When the gesture auto-created its paint layer on pointer-down, the created
   * layer contract (and where it was inserted). The engine composes this into
   * the stroke's history entry so an undo removes BOTH the stroke and the
   * now-empty auto-created layer (and a redo re-adds the layer + stroke).
   * Absent for strokes painted into a pre-existing layer.
   */
  createdLayer?: { layer: CanvasLayerContract; index: number };
}

export interface PixelEditPatch {
  rect: Rect;
  before: ImageData;
  after: ImageData;
}

export interface ControlPixelEditTransaction {
  readonly layerId: string;
  commitPatch(label: string, patch: PixelEditPatch): void;
  commitStroke(event: StrokeCommittedEvent): void;
  cancel(): void;
}

/**
 * A transient per-layer transform override the compositor/overlay read at render
 * time (a live drag preview that never touches the mirror). The move tool sets
 * only `x`/`y` (rotation/scale fall back to the committed transform); the
 * transform tool sets the full transform so a scale/rotate preview renders.
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
  getDocument(): CanvasDocumentContractV2 | null;
  /** Requests a re-render for the given flags. */
  invalidate(payload: InvalidatePayload): void;
  /** Reducer bridge. Painting tools use it for the single gesture-start `addCanvasLayer`. */
  dispatch(action: CanvasProjectMutation): void;
  /**
   * Records a structural document edit on the engine-owned canvas history:
   * dispatches `forward` now, and an undo dispatches `inverse` / a redo
   * re-dispatches `forward`. The move tool commits a layer nudge through this.
   */
  commitStructural(label: string, forward: CanvasProjectMutation, inverse: CanvasProjectMutation): void;
  /**
   * Sets (or clears with `null`) a transient per-layer transform override the
   * compositor and overlay read at render time — a live drag preview that never
   * touches the mirror/document. Cleared on commit or cancel.
   */
  setLayerTransformOverride(layerId: string, override: LayerTransformOverride | null): void;
  /**
   * Begins a transform session on `layerId` (captures its committed transform,
   * shows the live preview). Provided by the engine; the transform tool calls it
   * on activate / when a layer is clicked. Absent in minimal test harnesses.
   */
  beginTransformSession?(layerId: string): void;
  /** Prepares direct or transactional pixel editing for a selected control layer. */
  beginControlPixelEdit?(layerId: string): ControlPixelEditTransaction | null;
  /** Updates the active transform session's live transform (drag or numeric edit). */
  updateTransformSession?(transform: LayerTransform): void;
  /**
   * Commits the active transform session: a param commit (image layers) or a
   * pixel bake (paint layers), as ONE undoable entry. Then clears the session.
   */
  applyTransform?(): void;
  /** Cancels the active transform session (drops the preview, no dispatch). */
  cancelTransform?(): void;
  /**
   * Opens a CREATE-mode text-editing session at `docPoint` (no layer yet; the
   * commit later dispatches one `addCanvasLayer`). Seeds style from the text
   * options store. The text tool calls it on an empty-area click. Absent in
   * minimal test harnesses.
   */
  openTextCreate?(docPoint: Vec2): void;
  /**
   * Opens an EDIT-mode text-editing session on an existing text layer (captures
   * its committed source for the undo inverse). The text tool calls it when a
   * click hits a text layer. Absent in minimal test harnesses.
   */
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
  /**
   * Re-evaluates the active tool's CSS cursor and applies it to the input
   * element. A tool calls this when its `cursor(ctx)` result changes off a plain
   * pointer-move (e.g. the bbox tool switching to a resize cursor while hovering a
   * handle) — pointer-move does not otherwise refresh the cursor.
   */
  updateCursor(): void;
  /** Emits a completed-stroke event to `engine.tools.onStrokeCommitted` subscribers. */
  emitStrokeCommitted(event: StrokeCommittedEvent): void;
  /** Bumps a layer's cache version (without marking it stale) after a direct paint, and recomposites. */
  notifyLayerPainted(layerId: string): void;
  /**
   * Commits a lasso path to the engine's transient selection (boolean op applied
   * to the mask). Provided by the engine; the lasso tool calls it on pointer-up.
   * Absent in minimal test harnesses.
   */
  commitSelection?(commit: SelectionCommit): void;
  /**
   * The current selection mask as a placed surface (alpha 255 inside) in document
   * space — the mask is bounded to the selection extent, so its `rect` records
   * where it sits. `null` when there is no selection. Painting tools read it ONCE
   * on pointer-down to clip the stroke; a `null` result keeps the zero-overhead
   * hot path. Absent in minimal test harnesses.
   */
  getSelectionMask?(): PlacedSurface | null;
  /** Updates visual SAM input for the active engine-owned Select Object session. */
  updateSamInput?(input: SamVisualInput): void;
}

/**
 * Why a tool is being (de)activated, passed by the engine's `setTool` so a
 * session-bearing tool (transform) can tell a temporary modifier-hold switch
 * (space→view, alt→colorPicker; the pointer pipeline restores the prior tool
 * on release) apart from a REAL tool switch. A temp switch must not tear down
 * an in-progress session — only a real switch (or dispose) does.
 */
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
   * A session-level key command routed from the pointer pipeline: Enter →
   * `'apply'`, Escape → `'cancel'`. Tools with a multi-gesture session (transform)
   * use it to commit/abort; other tools ignore it. Escape also runs the normal
   * gesture cancel independently.
   */
  onKeyCommand?(ctx: ToolContext, command: 'apply' | 'cancel'): void;
  /** Wheel over the canvas; `screenAnchor` is the CSS-pixel cursor position. */
  onWheel?(ctx: ToolContext, deltaY: number, screenAnchor: { x: number; y: number }, modifiers: PointerModifiers): void;
  /** The CSS cursor to show while this tool is active. */
  cursor?(ctx: ToolContext): string;
}
