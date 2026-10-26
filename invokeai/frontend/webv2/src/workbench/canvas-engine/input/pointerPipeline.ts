/**
 * Normalizes DOM events into tool input with capture, coalesced samples and default mouse pressure 0.5. Middle
 * mouse pans independently. Space, Alt and held C temporarily select view, picker and bbox; release restores the
 * tool, while quick C selects bbox persistently. Temporary switches preserve sessions and are blocked mid-gesture.
 * Escape/pointercancel cancel; extra buttons are ignored mid-gesture. No key acts over an editable field, a
 * keyboard-navigated control, an open overlay or beneath a modal dialog; past that, the host's key-owner judge decides:
 * Enter needs the surface itself, the Escape ladder and C also accept a companion widget, and Space and Alt act over
 * the hovered canvas whatever the owner. Owned Space and Alt are prevented, and so is the release of an Alt the
 * pipeline acted on; other releases are observed, never consumed. DOM access is injected.
 */

import type { CanvasKeyOwner } from '@workbench/canvas-engine/capabilities';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { isModalPresent } from '@platform/ui/modalPresence';

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
  /** The focusable element whose focus gives the canvas its session keys. */
  getKeyboardRoot(): Pick<HTMLElement, 'contains'> | null;
  /**
   * The host's judge of which region owns a keydown; asked only on keydown, after the pipeline's own refusals. Absent
   * (tests, detached engines), focus alone decides: the keyboard root or no focus is the surface, and a control the
   * pointer last clicked lends its keys like a companion.
   */
  getKeyOwnerResolver?(): ((event: KeyboardEvent) => CanvasKeyOwner) | null;
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
  /** True while an undo/redo replays; primary presses start no gesture then (panning stays available). */
  isReplaying(): boolean;
}

/** The pipeline handle: DOM handlers plus lifecycle reset. */
export interface PointerPipeline {
  onPointerDown(event: PointerEvent): void;
  onPointerMove(event: PointerEvent): void;
  onPointerUp(event: PointerEvent): void;
  onPointerCancel(event: PointerEvent): void;
  /** The canvas gained hover; an Alt still held consumes a pending Alt hold. */
  onPointerEnter(event?: Pick<PointerEvent, 'altKey'>): void;
  onPointerLeave(): void;
  onKeyDown(event: KeyboardEvent): void;
  onKeyUp(event: KeyboardEvent): void;
  /** Window focus moves; records whether focus arrived by pointer, which decides hold-key ownership. */
  onFocusIn(event: FocusEvent): void;
  /** Primary gesture state blocks undo/redo from injecting pixels during live strokes. */
  isGestureActive(): boolean;
  /**
   * Where the pointer last moved over the canvas, in CSS pixels, including moves a middle-button pan consumes; null
   * once it left (outside a gesture) or the pipeline reset.
   */
  hoverPoint(): Vec2 | null;
  /**
   * Cancels capture/tool state and refreshes the cursor. Used on document replacement so a later release cannot
   * commit stale drag state; idle calls do nothing.
   */
  cancelActiveGesture(): void;
  /**
   * Before a genuine tool switch, cancels through the outgoing tool and ends any held temporary tool, so the switch
   * leaves the tool the user was really on and a later key release restores nothing.
   */
  endForToolSwitch(): void;
  /** Replaces a matching tool id that a currently-held temporary tool would restore on release. */
  replaceTemporaryRestoreTool(current: ToolId, replacement: ToolId): void;
  /** The tool a held temporary switch restores on release, i.e. the tool the user is really on; null without one. */
  heldRestoreToolId(): ToolId | null;
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

/** Open layers whose focused content keeps its own keys. */
const OVERLAY_SELECTOR = '[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]';

type KeyTarget = {
  tagName?: unknown;
  closest?: (selector: string) => unknown;
  matches?: (selector: string) => boolean;
} | null;

// Chromium turns `:focus-visible` on for the focused element at any keydown, so how focus arrived is read on arrival.
const arrivedByPointer = (target: EventTarget | null): boolean => {
  const el = target as KeyTarget;
  return typeof el?.matches === 'function' && !el.matches(':focus-visible');
};

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
  let hoverPoint: Vec2 | null = null;
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
  // A cancelled gesture's pointer keeps reporting pressed moves; they reach a tool only as hover until release.
  let cancelledPointerId: number | null = null;
  // The control focus last landed on by pointer; keyboard navigation (`:focus-visible` on arrival) clears it.
  // Held weakly: a focused row may be removed (virtualized) without another focusin.
  let pointerFocused: WeakRef<EventTarget> | null = null;
  // Alt went down while the surface owned it but the pointer was elsewhere: the next hover or press with Alt still
  // held begins the picker hold, so a pen that lands without hovering samples on contact.
  let altPending = false;
  // This pipeline began or armed a hold for the Alt now down; its release is prevented too.
  let altClaimed = false;

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

  const buildBatch = (event: PointerEvent, origin = inputOrigin()): PointerInput[] => {
    const coalesced = event.getCoalescedEvents?.();
    if (coalesced && coalesced.length > 0) {
      return coalesced.map((sample) => buildPointerInput(sample, origin));
    }
    return [buildPointerInput(event, origin)];
  };

  /** Enter belongs to the canvas only when focus is on its keyboard root (or nowhere) and no control claims it. */
  const focusOnSurface = (event: KeyboardEvent): boolean =>
    !isInteractiveTarget(event.target) &&
    (isDocumentRootTarget(event.target) || deps.getKeyboardRoot()?.contains(event.target as Node) === true);

  /**
   * Who owns a keydown the canvas might act on, or null when it is refused outright: already handled, under a modal
   * dialog (focus can rest on the body beneath one), over an editable field or an open overlay, or on a control
   * reached by keyboard, whose own activation must win.
   */
  const keyOwner = (event: KeyboardEvent): CanvasKeyOwner | null => {
    const target = event.target as KeyTarget;
    if (
      event.defaultPrevented ||
      isModalPresent(event) ||
      isEditableTarget(event.target) ||
      target?.closest?.(OVERLAY_SELECTOR)
    ) {
      return null;
    }
    const clicked = event.target !== null && event.target === pointerFocused?.deref();
    if (isInteractiveTarget(event.target) && !clicked) {
      return null;
    }
    const resolve = deps.getKeyOwnerResolver?.();
    if (resolve) {
      return resolve(event);
    }
    if (focusOnSurface(event)) {
      return 'surface';
    }
    return clicked ? 'companion' : null;
  };

  /** Enter applies only from the surface itself, with focus on its keyboard root or nowhere. */
  const ownsEnter = (event: KeyboardEvent): boolean => focusOnSurface(event) && keyOwner(event) === 'surface';

  /** The Escape ladder and C act from the surface or a companion widget. */
  const ownsSessionKey = (event: KeyboardEvent): boolean => {
    const owner = keyOwner(event);
    return owner === 'surface' || owner === 'companion';
  };

  /** Space and Alt are pointer-led: any owner while the canvas is hovered, otherwise only the surface. */
  const holdOwner = (event: KeyboardEvent): CanvasKeyOwner | null => {
    const owner = keyOwner(event);
    return owner !== null && (hovered || owner === 'surface') ? owner : null;
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
      cancelledPointerId = activePointerId;
      activePointerId = null;
    }
    deps.getActiveTool()?.onPointerCancel?.(deps.getToolContext());
    deps.updateCursor();
    if (restoreTempAfterGesture && tempHold) {
      endTempTool();
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
      endTempTool();
      deps.setTool(BBOX_TEMP_TOOL);
      return;
    }
    endTempTool();
  };

  return {
    cancelActiveGesture: () => {
      cancelGesture();
    },
    endForToolSwitch: () => {
      altPending = false;
      cancelGesture();
      if (tempHold) {
        endTempTool();
      }
    },
    onKeyDown: (event) => {
      if (event.key === 'Escape') {
        // Cancel the gesture before the engine Escape ladder. Editable fields keep Escape; gesture-consuming
        // Escape may cancel a session but must preserve committed selection.
        const gestureWasActive = gestureActive;
        cancelGesture();
        if (ownsSessionKey(event)) {
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
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && ownsEnter(event)) {
          deps.getActiveTool()?.onKeyCommand?.(deps.getToolContext(), 'apply');
        }
        return;
      }
      if (event.code === 'Space' && !event.repeat) {
        if (event.ctrlKey || event.metaKey || event.altKey || holdOwner(event) === null) {
          return;
        }
        beginTempTool('space', SPACE_TEMP_TOOL);
        if (tempHold === 'space') {
          event.preventDefault();
        }
        return;
      }
      if (isAltKey(event) && !event.repeat) {
        if (holdOwner(event) === null) {
          return;
        }
        // Firefox and Windows activate the menu bar on a lone Alt unless both its press and release are prevented.
        event.preventDefault();
        altClaimed = true;
        // A tool that reads Alt as a gesture modifier keeps it; pointer events still carry `altKey`.
        if (deps.getActiveTool()?.usesAltKey) {
          return;
        }
        if (hovered) {
          beginTempTool('alt', ALT_TEMP_TOOL);
        } else if (!tempHold && !gestureActive) {
          altPending = true;
        }
        return;
      }
      if (isBboxKey(event) && !event.repeat && ownsSessionKey(event)) {
        beginBboxTempTool();
      }
    },
    heldRestoreToolId: () => (tempHold && tempSwitched ? priorToolId : null),
    hoverPoint: () => hoverPoint,
    isGestureActive: () => gestureActive,
    onFocusIn: (event) => {
      pointerFocused = event.target && arrivedByPointer(event.target) ? new WeakRef(event.target) : null;
    },
    onKeyUp: (event) => {
      if (event.code === 'Space' && tempHold === 'space') {
        releaseTempTool();
      } else if (isAltKey(event)) {
        altPending = false;
        if (altClaimed) {
          altClaimed = false;
          event.preventDefault();
        }
        if (tempHold === 'alt') {
          releaseTempTool();
        }
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
      if (deps.isReplaying()) {
        event.preventDefault();
        return;
      }
      cancelledPointerId = null;
      if (altPending) {
        // The press is over the canvas whether or not it hovered first; begin the hold so the pick routes it.
        altPending = false;
        hovered = true;
        if (event.altKey) {
          beginTempTool('alt', ALT_TEMP_TOOL);
        }
      }
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
    onPointerEnter: (event) => {
      hovered = true;
      if (altPending) {
        altPending = false;
        if (event?.altKey) {
          beginTempTool('alt', ALT_TEMP_TOOL);
        }
      }
    },
    onPointerLeave: () => {
      hovered = false;
      if (!gestureActive) {
        hoverPoint = null;
      }
    },
    onPointerMove: (event) => {
      // Ignore move events from a pointer other than the one driving the active gesture/pan.
      if (activePointerId !== null && event.pointerId !== activePointerId) {
        return;
      }
      const origin = inputOrigin();
      hoverPoint = { x: event.clientX - origin.left, y: event.clientY - origin.top };
      if (middlePanning && middleLast) {
        const screenPoint = hoverPoint;
        deps.viewport.panBy({ x: screenPoint.x - middleLast.x, y: screenPoint.y - middleLast.y });
        middleLast = screenPoint;
        return;
      }
      // A cancelled gesture's pointer stays pressed until release; its moves reach the tool only as hover, so the
      // cursor keeps tracking but nothing resumes the gesture.
      const cancelledPress = cancelledPointerId === event.pointerId && event.buttons !== 0;
      if (cancelledPointerId === event.pointerId && !cancelledPress) {
        cancelledPointerId = null;
      }
      const batch = cancelledPress
        ? buildBatch(event, origin).map((input) => ({ ...input, buttons: 0 }))
        : buildBatch(event, origin);
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
      altPending = false;
      altClaimed = false;
      hovered = false;
      hoverPoint = null;
      gestureActive = false;
      activePointerId = null;
      cancelledPointerId = null;
      middlePanning = false;
      middleLast = null;
    },
  };
};
