/**
 * Normalizes DOM events into tool input with capture, coalesced samples and default mouse pressure 0.5. Middle
 * mouse pans independently. Space, Alt and held C temporarily select view, picker and bbox; release restores the
 * tool, while quick C selects bbox persistently. Temporary switches preserve sessions and are blocked mid-gesture.
 * Escape/pointercancel cancel; extra buttons are ignored mid-gesture. Enter and Space act only while the canvas
 * surface owns the keyboard; key releases are always observed and never consumed. DOM access is injected.
 */

import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

/** The tool id temporarily activated while alt is held (ships in Task P2.4). */
const ALT_TEMP_TOOL: ToolId = 'colorPicker';
/** The tool id temporarily activated while space is held. */
const SPACE_TEMP_TOOL: ToolId = 'view';
/** The tool id temporarily activated while the bbox key is held. */
const BBOX_TEMP_TOOL: ToolId = 'bbox';
/** Key-up before this threshold keeps the bbox tool selected, preserving the old tap behavior. */
const BBOX_QUICK_TAP_MS = 180;

/** Dependencies injected by the engine. */
export interface PointerPipelineDeps {
  viewport: Viewport;
  /** The element that owns pointer capture and defines the coordinate rect. */
  getInputElement(): (HTMLElement & Partial<Pick<HTMLElement, 'setPointerCapture' | 'releasePointerCapture'>>) | null;
  getActiveTool(): Tool | undefined;
  getActiveToolId(): ToolId;
  getToolContext(): ToolContext;
  /**
   * Temporary modifier switches and restores preserve tool sessions; see {@link beginTempTool} and {@link
   * endTempTool}.
   */
  setTool(id: ToolId, opts?: { temporary?: boolean }): void;
  hasTool(id: ToolId): boolean;
  updateCursor(): void;
  /**
   * Optional Escape handler after gesture cancellation, skipped in editable fields. `gestureWasActive` allows
   * session teardown while preserving committed selection.
   */
  handleEscape?(opts: { gestureWasActive: boolean }): void;
  /**
   * Optional primary-press hook before gesture activation. True consumes the press to commit a modal text session
   * without capture or tool routing, avoiding mid-gesture commit guards.
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
  /** Primary gesture state blocks undo/redo from injecting pixels during live strokes. */
  isGestureActive(): boolean;
  /**
   * Cancels capture/tool state and refreshes the cursor. Used on document replacement so a later release cannot
   * commit stale drag state; idle calls do nothing.
   */
  cancelActiveGesture(): void;
  /**
   * Cancels through the outgoing tool before a genuine tool switch. A hold whose key was released mid-gesture is
   * dropped instead of restored, so the requested tool wins.
   */
  cancelGestureForToolSwitch(): void;
  /** Replaces a matching tool id that a currently-held temporary tool would restore on release. */
  replaceTemporaryRestoreTool(current: ToolId, replacement: ToolId): void;
  /** Clears hover/gesture/temp-tool state, cancelling any in-flight gesture (called on detach/blur). */
  reset(): void;
}

const toPointerType = (type: string): PointerInput['pointerType'] =>
  type === 'pen' ? 'pen' : type === 'touch' ? 'touch' : 'mouse';

const isAltKey = (event: KeyboardEvent): boolean => event.code === 'AltLeft' || event.code === 'AltRight';
const isBboxKey = (event: KeyboardEvent): boolean =>
  event.code === 'KeyC' && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey;

/**
 * Editable targets retain their keys instead of activating temporary tools. Duck typing keeps this safe without
 * HTMLElement globals.
 */
const isEditableTarget = (target: EventTarget | null): boolean => {
  const el = target as { tagName?: unknown; isContentEditable?: unknown } | null;
  if (!el) {
    return false;
  }
  const tagName = typeof el.tagName === 'string' ? el.tagName.toUpperCase() : '';
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || el.isContentEditable === true;
};

/** Controls whose own keyboard activation must win over canvas session keys. */
const INTERACTIVE_SELECTOR = [
  'a[href]',
  'button',
  'input',
  'select',
  'summary',
  'textarea',
  '[contenteditable]:not([contenteditable="false"])',
  ...[
    'button',
    'checkbox',
    'combobox',
    'gridcell',
    'link',
    'menuitem',
    'menuitemcheckbox',
    'menuitemradio',
    'option',
    'radio',
    'slider',
    'spinbutton',
    'switch',
    'tab',
    'textbox',
    'treeitem',
  ].map((role) => `[role="${role}"]`),
].join(', ');

type KeyTarget = { tagName?: unknown; closest?: (selector: string) => unknown } | null;

const isInteractiveTarget = (target: EventTarget | null): boolean => {
  const el = target as KeyTarget;
  return isEditableTarget(target) || (typeof el?.closest === 'function' && el.closest(INTERACTIVE_SELECTOR) !== null);
};

/** A document root target means no control holds focus, so the canvas may take the key. */
const isDocumentRootTarget = (target: EventTarget | null): boolean => {
  const tagName = (target as KeyTarget)?.tagName;
  return !target || tagName === 'BODY' || tagName === 'HTML';
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
  let tempHold: 'space' | 'alt' | 'bbox' | null = null;
  let priorToolId: ToolId = deps.getActiveToolId();
  let tempSwitched = false;
  let restoreTempAfterGesture = false;
  let bboxQuickTap = false;
  let bboxTapTimer: ReturnType<typeof setTimeout> | null = null;
  // A cancelled gesture's pointer keeps reporting pressed moves; none may reach a tool until it is released.
  let cancelledPointerId: number | null = null;

  /**
   * Reads the input element's viewport offset. Hoisted out of
   * {@link buildPointerInput} so a coalesced batch pays for one layout read
   * instead of one per sample — the whole batch comes from a single event, so
   * the element cannot have moved between its samples.
   */
  const inputOrigin = (): { left: number; top: number } =>
    deps.getInputElement()?.getBoundingClientRect() ?? { left: 0, top: 0 };

  const buildPointerInput = (event: PointerEvent, origin = inputOrigin()): PointerInput => {
    const screenPoint: Vec2 = { x: event.clientX - origin.left, y: event.clientY - origin.top };
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
    const origin = inputOrigin();
    const coalesced = event.getCoalescedEvents?.();
    if (coalesced && coalesced.length > 0) {
      return coalesced.map((sample) => buildPointerInput(sample, origin));
    }
    return [buildPointerInput(event, origin)];
  };

  /** Session keys belong to the canvas only when focus is on its surface (or nowhere) and no control claims them. */
  const ownsKeyboard = (event: KeyboardEvent): boolean => {
    if (event.defaultPrevented || isInteractiveTarget(event.target)) {
      return false;
    }
    const input = deps.getInputElement();
    const surface = (input?.parentElement ?? input) as { contains?: (node: unknown) => boolean } | null;
    return isDocumentRootTarget(event.target) || surface?.contains?.(event.target) === true;
  };

  const releaseCapture = (pointerId: number): void => {
    deps.getInputElement()?.releasePointerCapture?.(pointerId);
  };

  /** `restoreTemporary: false` lets a genuine tool switch win over a hold whose key was released mid-gesture. */
  const cancelGesture = (restoreTemporary = true): void => {
    if (!gestureActive) {
      return;
    }
    gestureActive = false;
    if (activePointerId !== null) {
      releaseCapture(activePointerId);
      cancelledPointerId = activePointerId;
      activePointerId = null;
    }
    deps.getActiveTool()?.onPointerCancel?.(deps.getToolContext());
    deps.updateCursor();
    if (restoreTempAfterGesture && tempHold) {
      if (restoreTemporary) {
        endTempTool();
      } else {
        clearTempHold();
      }
    }
  };

  const clearBboxTapTimer = (): void => {
    if (bboxTapTimer !== null) {
      clearTimeout(bboxTapTimer);
      bboxTapTimer = null;
    }
  };

  const markBboxHold = (): void => {
    if (tempHold !== 'bbox') {
      return;
    }
    bboxQuickTap = false;
    clearBboxTapTimer();
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

  const beginBboxTempTool = (): void => {
    if (tempHold || gestureActive) {
      return;
    }
    tempHold = 'bbox';
    priorToolId = deps.getActiveToolId();
    restoreTempAfterGesture = false;
    bboxQuickTap = true;
    clearBboxTapTimer();
    bboxTapTimer = setTimeout(() => {
      bboxQuickTap = false;
      bboxTapTimer = null;
    }, BBOX_QUICK_TAP_MS);

    if (hovered && deps.hasTool(BBOX_TEMP_TOOL)) {
      deps.setTool(BBOX_TEMP_TOOL, { temporary: true });
      tempSwitched = true;
    } else {
      tempSwitched = false;
    }
  };

  function clearTempHold(): void {
    clearBboxTapTimer();
    tempHold = null;
    tempSwitched = false;
    restoreTempAfterGesture = false;
    bboxQuickTap = false;
  }

  function endTempTool(): void {
    const restore = tempSwitched;
    clearTempHold();
    if (restore) {
      deps.setTool(priorToolId, { temporary: true });
    }
  }

  const releaseTempTool = (): void => {
    if (!tempHold) {
      return;
    }
    if (gestureActive) {
      restoreTempAfterGesture = true;
      return;
    }
    endTempTool();
  };

  const releaseBboxTempTool = (): void => {
    if (tempHold !== 'bbox') {
      return;
    }
    clearBboxTapTimer();
    if (gestureActive) {
      bboxQuickTap = false;
      restoreTempAfterGesture = true;
      return;
    }
    if (bboxQuickTap) {
      if (tempSwitched) {
        deps.setTool(priorToolId, { temporary: true });
      }
      deps.setTool(BBOX_TEMP_TOOL);
      tempHold = null;
      tempSwitched = false;
      restoreTempAfterGesture = false;
      bboxQuickTap = false;
      return;
    }
    endTempTool();
  };

  return {
    cancelActiveGesture: () => {
      cancelGesture();
    },
    cancelGestureForToolSwitch: () => {
      cancelGesture(false);
    },
    onKeyDown: (event) => {
      if (event.key === 'Escape') {
        // Cancel the gesture before the engine Escape ladder. Editable fields keep Escape; gesture-consuming
        // Escape may cancel a session but must preserve committed selection.
        const gestureWasActive = gestureActive;
        cancelGesture();
        if (!isEditableTarget(event.target)) {
          deps.handleEscape?.({ gestureWasActive });
        }
        return;
      }
      // Editable fields own temporary-tool and Enter keys.
      if (isEditableTarget(event.target)) {
        return;
      }
      if (event.key === 'Enter') {
        // Enter applies a session-bearing tool's edit (transform). No-op otherwise.
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && ownsKeyboard(event)) {
          deps.getActiveTool()?.onKeyCommand?.(deps.getToolContext(), 'apply');
        }
        return;
      }
      if (event.code === 'Space' && !event.repeat) {
        if (event.ctrlKey || event.metaKey || event.altKey || !ownsKeyboard(event)) {
          return;
        }
        beginTempTool('space', SPACE_TEMP_TOOL);
        if (tempHold === 'space') {
          event.preventDefault();
        }
        return;
      }
      if (isAltKey(event) && !event.repeat) {
        if (!deps.getActiveTool()?.usesAltKey) {
          beginTempTool('alt', ALT_TEMP_TOOL);
        }
        return;
      }
      if (isBboxKey(event) && !event.repeat) {
        beginBboxTempTool();
      }
    },
    isGestureActive: () => gestureActive,
    onKeyUp: (event) => {
      if (event.code === 'Space' && tempHold === 'space') {
        releaseTempTool();
      } else if (isAltKey(event) && tempHold === 'alt') {
        releaseTempTool();
      } else if (isBboxKey(event) && tempHold === 'bbox') {
        releaseBboxTempTool();
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
      cancelledPointerId = null;
      // Commit and consume modal text presses before gesture activation; prevent default focus/selection after
      // closing the session.
      if (deps.maybeCommitModalSession?.()) {
        event.preventDefault();
        return;
      }
      markBboxHold();
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
      if (cancelledPointerId === event.pointerId) {
        if (event.buttons !== 0) {
          return;
        }
        cancelledPointerId = null;
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
      if (cancelledPointerId === event.pointerId) {
        cancelledPointerId = null;
      }
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
      if (restoreTempAfterGesture && tempHold) {
        endTempTool();
      }
    },
    replaceTemporaryRestoreTool: (current, replacement) => {
      if (tempHold && priorToolId === current) {
        priorToolId = replacement;
      }
    },
    reset: () => {
      // Cancel through the tool that owns the gesture before a held temporary tool is restored.
      cancelGesture();
      if (tempHold) {
        endTempTool();
      }
      clearBboxTapTimer();
      hovered = false;
      gestureActive = false;
      activePointerId = null;
      cancelledPointerId = null;
      middlePanning = false;
      middleLast = null;
    },
  };
};
