/**
 * The pointer pipeline: normalizes raw DOM pointer/key events into the engine's
 * {@link PointerInput} vocabulary and routes them to the active tool.
 *
 * Responsibilities lifted out of the engine so `engine.ts` stays lean:
 * - Pointer capture on down; `getCoalescedEvents()` batching on move (so fast
 *   strokes keep every intermediate sample); mouse pressure defaulted to 0.5.
 * - Middle-mouse pan (engine-level, tool-independent).
 * - Modifier-hold temporary tools: space → view; alt → colorPicker (a no-op if
 *   that tool isn't registered yet — the registry decides). The prior tool is
 *   restored on release. Temp switches are suppressed mid-gesture, and are
 *   flagged `{ temporary: true }` on `setTool` so a session-bearing tool
 *   (transform) can tell them apart from a real switch and keep its session
 *   alive across the hold.
 * - Gesture cancellation: pointercancel and Esc route to the tool's
 *   `onPointerCancel`; secondary/extra buttons are ignored during a gesture.
 *
 * The pipeline reaches the DOM only through the injected `getInputElement`
 * (for pointer capture / element rect) and the events passed to its handlers, so
 * it is fully driveable by a fake harness in node tests. Zero React, zero
 * import-time side effects.
 */

import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

/** The tool id temporarily activated while alt is held (ships in Task P2.4). */
const ALT_TEMP_TOOL: ToolId = 'colorPicker';
/** The tool id temporarily activated while space is held. */
const SPACE_TEMP_TOOL: ToolId = 'view';

/** Dependencies injected by the engine. */
export interface PointerPipelineDeps {
  viewport: Viewport;
  /** The element that owns pointer capture and defines the coordinate rect. */
  getInputElement(): (HTMLElement & Partial<Pick<HTMLElement, 'setPointerCapture' | 'releasePointerCapture'>>) | null;
  getActiveTool(): Tool | undefined;
  getActiveToolId(): ToolId;
  getToolContext(): ToolContext;
  /**
   * Switches the active tool. `opts.temporary` marks a modifier-hold switch
   * (and its matching restore) so a session-bearing tool's `onActivate`/
   * `onDeactivate` can preserve its session instead of tearing it down — see
   * {@link beginTempTool} / {@link endTempTool}.
   */
  setTool(id: ToolId, opts?: { temporary?: boolean }): void;
  hasTool(id: ToolId): boolean;
  updateCursor(): void;
  /**
   * The engine's Escape priority, run AFTER the in-flight gesture is cancelled
   * (and skipped in editable fields): cancel a transform session, else deselect.
   * `gestureWasActive` tells it a drag just consumed this Escape, so it should
   * cancel a session (session teardown is wanted mid-drag, matching the prior
   * behavior) but NOT also deselect. Optional so minimal harnesses can omit it.
   * See `engine.ts` `handleEscape`.
   */
  handleEscape?(opts: { gestureWasActive: boolean }): void;
  /**
   * Called on a primary-button pointerdown BEFORE any gesture starts. Returns
   * `true` if the engine consumed the press to commit an open modal session (a
   * text-edit session), in which case the pipeline swallows the press entirely —
   * no capture, no gesture, no tool routing — because the click's sole job was to
   * close the session (the next press then starts a fresh interaction). Running
   * before `gestureActive` is set is what lets the commit through the engine's
   * mid-gesture guard. Optional so minimal harnesses can omit it.
   */
  maybeCommitModalSession?(): boolean;
}

/** The pipeline handle: DOM handlers plus lifecycle reset. */
export interface PointerPipeline {
  onPointerDown(event: PointerEvent): void;
  onPointerMove(event: PointerEvent): void;
  onPointerUp(event: PointerEvent): void;
  onPointerCancel(event: PointerEvent): void;
  onPointerEnter(): void;
  onPointerLeave(): void;
  onKeyDown(event: KeyboardEvent): void;
  onKeyUp(event: KeyboardEvent): void;
  /**
   * True while a primary-button paint/drag gesture is mid-stroke (pointer down,
   * not yet up/cancel). The engine consults this to no-op undo/redo during a
   * live stroke, so a mid-gesture mod+z can't inject pixels under the session.
   */
  isGestureActive(): boolean;
  /**
   * Cancels an in-flight primary-button gesture the same way Esc/pointercancel
   * do: releases pointer capture, runs the active tool's `onPointerCancel` (so it
   * drops its own transient state), and refreshes the cursor. A no-op when no
   * gesture is active. The engine calls this on a wholesale document replacement
   * so a mid-drag swap can't commit against the outgoing document on pointer-up.
   */
  cancelActiveGesture(): void;
  /** Clears hover/gesture/temp-tool state, cancelling any in-flight gesture (called on detach/blur). */
  reset(): void;
}

const toPointerType = (type: string): PointerInput['pointerType'] =>
  type === 'pen' ? 'pen' : type === 'touch' ? 'touch' : 'mouse';

const isAltKey = (event: KeyboardEvent): boolean => event.code === 'AltLeft' || event.code === 'AltRight';

/**
 * True when the key event targets an editable element (text input, textarea, or
 * a contenteditable node). The space/alt temp-tool holds are window-level, so
 * without this guard typing a space in a rename field would hijack the canvas
 * view tool. Duck-typed on `tagName`/`isContentEditable` so it stays node-safe
 * (no `instanceof HTMLElement`, which throws where those globals are absent).
 */
const isEditableTarget = (target: EventTarget | null): boolean => {
  const el = target as { tagName?: unknown; isContentEditable?: unknown } | null;
  if (!el) {
    return false;
  }
  const tagName = typeof el.tagName === 'string' ? el.tagName.toUpperCase() : '';
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || el.isContentEditable === true;
};

/** Creates a pointer pipeline bound to the engine's injected deps. */
export const createPointerPipeline = (deps: PointerPipelineDeps): PointerPipeline => {
  let hovered = false;
  // Primary-button paint/drag gesture in progress.
  let gestureActive = false;
  let activePointerId: number | null = null;
  // Middle-mouse pan.
  let middlePanning = false;
  let middleLast: Vec2 | null = null;
  // Temporary modifier-hold tool.
  let tempHold: 'space' | 'alt' | null = null;
  let priorToolId: ToolId = deps.getActiveToolId();
  let tempSwitched = false;

  const buildPointerInput = (event: PointerEvent): PointerInput => {
    const el = deps.getInputElement();
    const rect = el?.getBoundingClientRect() ?? ({ left: 0, top: 0 } as DOMRect);
    const screenPoint: Vec2 = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    return {
      buttons: event.buttons,
      documentPoint: deps.viewport.screenToDocument(screenPoint),
      modifiers: { alt: event.altKey, ctrl: event.ctrlKey, meta: event.metaKey, shift: event.shiftKey },
      pointerType: toPointerType(event.pointerType),
      pressure: event.pressure > 0 ? event.pressure : 0.5,
      screenPoint,
      timeStamp: event.timeStamp,
    };
  };

  const buildBatch = (event: PointerEvent): PointerInput[] => {
    const coalesced = event.getCoalescedEvents?.();
    if (coalesced && coalesced.length > 0) {
      return coalesced.map(buildPointerInput);
    }
    return [buildPointerInput(event)];
  };

  const releaseCapture = (pointerId: number): void => {
    deps.getInputElement()?.releasePointerCapture?.(pointerId);
  };

  const cancelGesture = (): void => {
    if (!gestureActive) {
      return;
    }
    gestureActive = false;
    if (activePointerId !== null) {
      releaseCapture(activePointerId);
      activePointerId = null;
    }
    deps.getActiveTool()?.onPointerCancel?.(deps.getToolContext());
    deps.updateCursor();
  };

  const beginTempTool = (hold: 'space' | 'alt', toolId: ToolId): void => {
    if (tempHold || gestureActive || !hovered) {
      return;
    }
    tempHold = hold;
    priorToolId = deps.getActiveToolId();
    if (deps.hasTool(toolId)) {
      deps.setTool(toolId, { temporary: true });
      tempSwitched = true;
    } else {
      tempSwitched = false;
    }
  };

  const endTempTool = (): void => {
    if (tempSwitched) {
      deps.setTool(priorToolId, { temporary: true });
    }
    tempHold = null;
    tempSwitched = false;
  };

  return {
    cancelActiveGesture: () => {
      cancelGesture();
    },
    onKeyDown: (event) => {
      if (event.key === 'Escape') {
        // Cancel any active drag first, then run the engine's Escape priority
        // (cancel a transform session, else deselect). The editable guard keeps
        // Escape in a text field from tearing down a canvas session/selection the
        // field isn't part of. A mid-drag Escape cancels the gesture here; the
        // subsequent `handleEscape` still sees (and cancels) a transform session
        // the reverted drag kept open, matching the prior gesture+session teardown,
        // but skips deselect so a mid-lasso Escape drops only the in-progress path.
        const gestureWasActive = gestureActive;
        cancelGesture();
        if (!isEditableTarget(event.target)) {
          deps.handleEscape?.({ gestureWasActive });
        }
        return;
      }
      // Never let space/alt temp-tool holds (or Enter apply) fire while the user is
      // typing in an editable field (e.g. a layer rename input) — that key belongs
      // to the field, not the canvas.
      if (isEditableTarget(event.target)) {
        return;
      }
      if (event.key === 'Enter') {
        // Enter applies a session-bearing tool's edit (transform). No-op otherwise.
        deps.getActiveTool()?.onKeyCommand?.(deps.getToolContext(), 'apply');
        return;
      }
      if (event.code === 'Space' && !event.repeat) {
        beginTempTool('space', SPACE_TEMP_TOOL);
        if (tempHold === 'space') {
          event.preventDefault();
        }
        return;
      }
      if (isAltKey(event) && !event.repeat) {
        beginTempTool('alt', ALT_TEMP_TOOL);
      }
    },
    isGestureActive: () => gestureActive,
    onKeyUp: (event) => {
      if (event.code === 'Space' && tempHold === 'space') {
        endTempTool();
      } else if (isAltKey(event) && tempHold === 'alt') {
        endTempTool();
      }
    },
    onPointerCancel: (event) => {
      // Ignore cancel events from a pointer other than the one driving the active gesture/pan
      // (pointer capture on the active pointer does not suppress other pointers' events).
      if (activePointerId !== null && event.pointerId !== activePointerId) {
        return;
      }
      if (middlePanning) {
        releaseCapture(event.pointerId);
        middlePanning = false;
        middleLast = null;
        activePointerId = null;
        return;
      }
      // `cancelGesture` releases the captured pointer itself.
      cancelGesture();
    },
    onPointerDown: (event) => {
      // Ignore extra/secondary buttons pressed during an active gesture or pan.
      if (gestureActive || middlePanning) {
        return;
      }
      const el = deps.getInputElement();
      if (event.button === 1) {
        el?.setPointerCapture?.(event.pointerId);
        activePointerId = event.pointerId;
        middlePanning = true;
        middleLast = buildPointerInput(event).screenPoint;
        return;
      }
      if (event.button !== 0) {
        return;
      }
      // A primary press while a text-edit session is open commits it (engine-side,
      // reading the live portal content) and is swallowed — no gesture, no tool
      // routing. This must precede `gestureActive = true` so the engine's
      // mid-gesture commit guard cannot drop it. `preventDefault` avoids a stray
      // focus/selection default now that the session has already been closed.
      if (deps.maybeCommitModalSession?.()) {
        event.preventDefault();
        return;
      }
      el?.setPointerCapture?.(event.pointerId);
      activePointerId = event.pointerId;
      gestureActive = true;
      event.preventDefault();
      deps.getActiveTool()?.onPointerDown?.(deps.getToolContext(), buildPointerInput(event));
      deps.updateCursor();
    },
    onPointerEnter: () => {
      hovered = true;
    },
    onPointerLeave: () => {
      hovered = false;
    },
    onPointerMove: (event) => {
      // Ignore move events from a pointer other than the one driving the active gesture/pan.
      if (activePointerId !== null && event.pointerId !== activePointerId) {
        return;
      }
      if (middlePanning && middleLast) {
        const screenPoint = buildPointerInput(event).screenPoint;
        deps.viewport.panBy({ x: screenPoint.x - middleLast.x, y: screenPoint.y - middleLast.y });
        middleLast = screenPoint;
        return;
      }
      const batch = buildBatch(event);
      const last = batch[batch.length - 1];
      if (!last) {
        return;
      }
      deps.getActiveTool()?.onPointerMove?.(deps.getToolContext(), last, batch);
    },
    onPointerUp: (event) => {
      // Ignore up events from a pointer other than the one driving the active gesture/pan.
      if (activePointerId !== null && event.pointerId !== activePointerId) {
        return;
      }
      releaseCapture(event.pointerId);
      if (middlePanning) {
        middlePanning = false;
        middleLast = null;
        activePointerId = null;
        return;
      }
      if (!gestureActive) {
        return;
      }
      gestureActive = false;
      activePointerId = null;
      deps.getActiveTool()?.onPointerUp?.(deps.getToolContext(), buildPointerInput(event));
      deps.updateCursor();
    },
    reset: () => {
      if (tempHold) {
        endTempTool();
      }
      // Cancel an in-flight gesture through the same path Esc uses, so the active
      // tool's `onPointerCancel` runs and clears its transient state (rather than
      // silently dropping `gestureActive` and stranding stale tool state).
      cancelGesture();
      hovered = false;
      gestureActive = false;
      activePointerId = null;
      middlePanning = false;
      middleLast = null;
    },
  };
};
