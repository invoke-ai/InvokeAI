/** Normalizes DOM events into tool input; Space, Alt and held C pick temporary tools, blocked mid-gesture.
 Keys never act over editable fields, keyboard-navigated controls, open overlays or modal dialogs. */

import type { CanvasKeyOwner } from '@workbench/canvas-engine/capabilities';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput, ToolId, Vec2 } from '@workbench/canvas-engine/types';
import type { Viewport } from '@workbench/canvas-engine/viewport';

import { isModalPresent } from '@platform/ui/modalPresence';

const ALT_TEMP_TOOL: ToolId = 'colorPicker';
const SPACE_TEMP_TOOL: ToolId = 'view';
const BBOX_TEMP_TOOL: ToolId = 'bbox';
const BBOX_QUICK_TAP_MS = 180;
/** Tool pressure for a pointer that reports none: a mouse, and any pointer while it hovers (the DOM reports 0). */
const UNREPORTED_PRESSURE = 0.5;

export interface PointerPipelineDeps {
  viewport: Viewport;
  getInputElement(): (HTMLElement & Partial<Pick<HTMLElement, 'setPointerCapture' | 'releasePointerCapture'>>) | null;
  getKeyboardRoot(): Pick<HTMLElement, 'contains'> | null;
  /** Absent, focus alone decides, and a control the pointer last clicked lends its keys like a companion. */
  getKeyOwnerResolver?(): ((event: KeyboardEvent) => CanvasKeyOwner) | null;
  getActiveTool(): Tool | undefined;
  getActiveToolId(): ToolId;
  getToolContext(): ToolContext;
  setTool(id: ToolId, opts?: { temporary?: boolean }): void;
  hasTool(id: ToolId): boolean;
  updateCursor(): void;
  /** `gestureWasActive` allows session teardown while preserving committed selection. */
  handleEscape?(opts: { gestureWasActive: boolean }): void;
  /** True consumes the press to commit a modal text session, avoiding mid-gesture commit guards. */
  maybeCommitModalSession?(): boolean;
  /** Primary presses start no gesture then; panning stays available. */
  isReplaying(): boolean;
  /** Middle-button pans do not count. */
  onGestureChange?(active: boolean): void;
  /**
   * Whether the document has focus. Absent (tests, detached engines), it is taken as focused. Pointer events still
   * reach a visible page while another app, the address bar or DevTools has focus, and their modifier state then says
   * nothing about a Control press the page cannot see.
   */
  hasFocus?(): boolean;
}

/** Whether the physical Control key is down, as far as the pipeline has seen. */
export type ControlKeyState = 'down' | 'up' | 'unknown';

export interface PointerPipeline {
  onPointerDown(event: PointerEvent): void;
  onPointerMove(event: PointerEvent): void;
  onPointerUp(event: PointerEvent): void;
  onPointerCancel(event: PointerEvent): void;
  /** An Alt still held consumes a pending Alt hold. */
  onPointerEnter(event?: Pick<PointerEvent, 'altKey'>): void;
  onPointerLeave(): void;
  onKeyDown(event: KeyboardEvent): void;
  onKeyUp(event: KeyboardEvent): void;
  /** Capture-phase window key events, so a handler that stops propagation cannot hide Control. */
  onKeyCapture(event: KeyboardEvent): void;
  /** A Control press or release made while the window was away went unseen. */
  onWindowFocus(): void;
  /** Records whether focus arrived by pointer, which decides hold-key ownership. */
  onFocusIn(event: FocusEvent): void;
  isGestureActive(): boolean;
  /** CSS pixels, including middle-pan moves; null once it left outside a gesture. */
  hoverPoint(): Vec2 | null;
  /** `unknown` until a key or focused pointer event shows it, and again whenever focus may have changed hands. */
  controlKeyState(): ControlKeyState;
  /** The view moved under a still pointer: re-delivers a hover move so hover state follows. Skipped mid-gesture. */
  refreshHover(): void;
  /** Used on document replacement so a later release cannot commit stale drag state. */
  cancelActiveGesture(): void;
  /** Ends any held temporary tool, so a later key release restores nothing. */
  endForToolSwitch(): void;
  replaceTemporaryRestoreTool(current: ToolId, replacement: ToolId): void;
  /** Null without a held temporary tool. */
  heldRestoreToolId(): ToolId | null;
  reset(): void;
}

const toPointerType = (type: string): PointerInput['pointerType'] =>
  type === 'pen' ? 'pen' : type === 'touch' ? 'touch' : 'mouse';

const isAltKey = (event: KeyboardEvent): boolean => event.code === 'AltLeft' || event.code === 'AltRight';
const isBboxKey = (event: KeyboardEvent): boolean =>
  event.code === 'KeyC' && !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey;

/** Duck typing avoids HTMLElement globals. */
const isEditableTarget = (target: EventTarget | null): boolean => {
  const el = target as { tagName?: unknown; isContentEditable?: unknown } | null;
  if (!el) {
    return false;
  }
  const tagName = typeof el.tagName === 'string' ? el.tagName.toUpperCase() : '';
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || el.isContentEditable === true;
};

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

const OVERLAY_SELECTOR = '[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]';

type KeyTarget = {
  tagName?: unknown;
  closest?: (selector: string) => unknown;
  matches?: (selector: string) => boolean;
} | null;

// Chromium turns `:focus-visible` on at any keydown, so how focus arrived is read on arrival.
const arrivedByPointer = (target: EventTarget | null): boolean => {
  const el = target as KeyTarget;
  return typeof el?.matches === 'function' && !el.matches(':focus-visible');
};

const isInteractiveTarget = (target: EventTarget | null): boolean => {
  const el = target as KeyTarget;
  return isEditableTarget(target) || (typeof el?.closest === 'function' && el.closest(INTERACTIVE_SELECTOR) !== null);
};

const isDocumentRootTarget = (target: EventTarget | null): boolean => {
  const tagName = (target as KeyTarget)?.tagName;
  return !target || tagName === 'BODY' || tagName === 'HTML';
};

export const createPointerPipeline = (deps: PointerPipelineDeps): PointerPipeline => {
  let hovered = false;
  let hoverPoint: Vec2 | null = null;
  let controlKey: ControlKeyState = 'unknown';
  // The last move a tool received, so a hover refresh keeps its pointer type and modifiers.
  let lastToolMove: PointerInput | null = null;
  let gestureActive = false;
  let activePointerId: number | null = null;
  let middlePanning = false;
  let middleLast: Vec2 | null = null;
  let tempHold: 'space' | 'alt' | 'bbox' | null = null;
  let priorToolId: ToolId = deps.getActiveToolId();
  let tempSwitched = false;
  let restoreTempAfterGesture = false;
  let bboxQuickTap = false;
  let bboxTapTimer: ReturnType<typeof setTimeout> | null = null;
  // A cancelled gesture's pointer keeps reporting pressed moves; they reach a tool only as hover until release.
  let cancelledPointerId: number | null = null;
  // Weak: a focused row may be removed (virtualized) without another focusin.
  let pointerFocused: WeakRef<EventTarget> | null = null;
  // Alt pressed away from the canvas: the next hover or press with Alt still held begins the picker hold.
  let altPending = false;
  // This pipeline acted on the Alt now down, so its release is prevented too.
  let altClaimed = false;

  /** One layout read per coalesced batch; the element cannot move between a single event's samples. */
  const inputOrigin = (): { left: number; top: number } =>
    deps.getInputElement()?.getBoundingClientRect() ?? { left: 0, top: 0 };

  /** Control as a pointer event shows it; see `controlKeyState`. */
  const observePointerControl = (event: PointerEvent): void => {
    controlKey = event.ctrlKey ? 'down' : deps.hasFocus?.() === false ? 'unknown' : 'up';
  };

  const buildPointerInput = (event: PointerEvent, origin = inputOrigin()): PointerInput => {
    const screenPoint: Vec2 = { x: event.clientX - origin.left, y: event.clientY - origin.top };
    return {
      buttons: event.buttons,
      documentPoint: deps.viewport.screenToDocument(screenPoint),
      modifiers: { alt: event.altKey, ctrl: event.ctrlKey, meta: event.metaKey, shift: event.shiftKey },
      pointerType: toPointerType(event.pointerType),
      pressure: event.pressure > 0 ? event.pressure : UNREPORTED_PRESSURE,
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

  const focusOnSurface = (event: KeyboardEvent): boolean =>
    !isInteractiveTarget(event.target) &&
    (isDocumentRootTarget(event.target) || deps.getKeyboardRoot()?.contains(event.target as Node) === true);

  /** Null when refused outright; focus can rest on the body beneath a modal dialog. */
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

  const ownsEnter = (event: KeyboardEvent): boolean => focusOnSurface(event) && keyOwner(event) === 'surface';

  const ownsSessionKey = (event: KeyboardEvent): boolean => {
    const owner = keyOwner(event);
    return owner === 'surface' || owner === 'companion';
  };

  /** Pointer-led: any owner while the canvas is hovered, otherwise only the surface. */
  const holdOwner = (event: KeyboardEvent): CanvasKeyOwner | null => {
    const owner = keyOwner(event);
    return owner !== null && (hovered || owner === 'surface') ? owner : null;
  };

  const setGestureActive = (active: boolean): void => {
    if (gestureActive !== active) {
      gestureActive = active;
      deps.onGestureChange?.(active);
    }
  };

  const releaseCapture = (pointerId: number): void => {
    deps.getInputElement()?.releasePointerCapture?.(pointerId);
  };

  const cancelGesture = (): void => {
    if (!gestureActive) {
      return;
    }
    setGestureActive(false);
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
    onKeyCapture: (event) => {
      // Browsers may still report ctrlKey on the Control key's own release.
      const released = event.type === 'keyup' && event.key === 'Control';
      controlKey = !released && (event.key === 'Control' || event.ctrlKey) ? 'down' : 'up';
    },
    onKeyDown: (event) => {
      if (event.key === 'Escape') {
        const gestureWasActive = gestureActive;
        cancelGesture();
        if (ownsSessionKey(event)) {
          deps.handleEscape?.({ gestureWasActive });
        }
        return;
      }
      if (isEditableTarget(event.target)) {
        return;
      }
      if (event.key === 'Enter') {
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && ownsEnter(event)) {
          deps.getActiveTool()?.onKeyCommand?.(deps.getToolContext(), 'apply');
        }
        return;
      }
      if (event.code === 'Space' && !event.repeat) {
        if (event.ctrlKey || event.metaKey || holdOwner(event) === null) {
          return;
        }
        if (deps.getActiveTool()?.onModifierKey?.(deps.getToolContext(), 'Space', true)) {
          event.preventDefault();
          return;
        }
        if (event.altKey) {
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
        // Firefox and Windows activate the menu bar on a lone Alt unless both press and release are prevented.
        event.preventDefault();
        altClaimed = true;
        if (deps.getActiveTool()?.onModifierKey?.(deps.getToolContext(), 'Alt', true)) {
          return;
        }
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
      if (event.key === 'Shift' && holdOwner(event) !== null) {
        deps.getActiveTool()?.onModifierKey?.(deps.getToolContext(), 'Shift', true);
      }
    },
    controlKeyState: () => controlKey,
    heldRestoreToolId: () => (tempHold && tempSwitched ? priorToolId : null),
    hoverPoint: () => hoverPoint,
    isGestureActive: () => gestureActive,
    onFocusIn: (event) => {
      pointerFocused = event.target && arrivedByPointer(event.target) ? new WeakRef(event.target) : null;
    },
    onKeyUp: (event) => {
      const modifier =
        event.code === 'Space' ? 'Space' : isAltKey(event) ? 'Alt' : event.key === 'Shift' ? 'Shift' : null;
      if (modifier) {
        deps.getActiveTool()?.onModifierKey?.(deps.getToolContext(), modifier, false);
      }
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
      // Pointer capture does not suppress other pointers' events.
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
      cancelGesture();
    },
    onPointerDown: (event) => {
      observePointerControl(event);
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
        altPending = false;
        hovered = true;
        if (event.altKey) {
          beginTempTool('alt', ALT_TEMP_TOOL);
        }
      }
      if (deps.maybeCommitModalSession?.()) {
        event.preventDefault();
        return;
      }
      markBboxHold();
      el?.setPointerCapture?.(event.pointerId);
      activePointerId = event.pointerId;
      setGestureActive(true);
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
      if (activePointerId !== null && event.pointerId !== activePointerId) {
        return;
      }
      observePointerControl(event);
      const origin = inputOrigin();
      hoverPoint = { x: event.clientX - origin.left, y: event.clientY - origin.top };
      if (middlePanning && middleLast) {
        const screenPoint = hoverPoint;
        deps.viewport.panBy({ x: screenPoint.x - middleLast.x, y: screenPoint.y - middleLast.y });
        middleLast = screenPoint;
        return;
      }
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
      lastToolMove = last;
      deps.getActiveTool()?.onPointerMove?.(deps.getToolContext(), last, batch);
    },
    onPointerUp: (event) => {
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
      setGestureActive(false);
      activePointerId = null;
      deps.getActiveTool()?.onPointerUp?.(deps.getToolContext(), buildPointerInput(event));
      deps.updateCursor();
      if (restoreTempAfterGesture && tempHold) {
        endTempTool();
      }
    },
    onWindowFocus: () => {
      controlKey = 'unknown';
    },
    replaceTemporaryRestoreTool: (current, replacement) => {
      if (tempHold && priorToolId === current) {
        priorToolId = replacement;
      }
    },
    refreshHover: () => {
      if (!hovered || !hoverPoint || !lastToolMove || gestureActive || middlePanning) {
        return;
      }
      const input: PointerInput = {
        ...lastToolMove,
        buttons: 0,
        documentPoint: deps.viewport.screenToDocument(hoverPoint),
        pressure: UNREPORTED_PRESSURE,
        screenPoint: hoverPoint,
      };
      deps.getActiveTool()?.onPointerMove?.(deps.getToolContext(), input, [input]);
    },
    reset: () => {
      cancelGesture();
      if (tempHold) {
        endTempTool();
      }
      clearBboxTapTimer();
      // Tool-owned holds can outlive a press (polygon drafts) and miss keyup while the window is unfocused.
      const tool = deps.getActiveTool();
      const ctx = deps.getToolContext();
      for (const modifier of ['Space', 'Alt', 'Shift'] as const) {
        tool?.onModifierKey?.(ctx, modifier, false);
      }
      altPending = false;
      altClaimed = false;
      hovered = false;
      hoverPoint = null;
      lastToolMove = null;
      controlKey = 'unknown';
      gestureActive = false;
      activePointerId = null;
      cancelledPointerId = null;
      middlePanning = false;
      middleLast = null;
    },
  };
};
