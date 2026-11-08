import { MouseSensor, type Activator, type SensorOptions, type SensorProps } from '@dnd-kit/core';

/**
 * For touch on pannable surfaces, hold first, then require movement: early movement yields to scrolling and
 * motionless holds remain taps. Pen and touch-action:none surfaces activate by distance, unless the surface marks
 * itself with `data-drag-hold-on-touch` because it owns one-finger gestures of its own (the preview swipe); early
 * movement then yields to that gesture. Once armed, a non-passive touchmove listener prevents native pan from
 * stealing the gesture. Every exit, including pointercancel, calls onEnd/onCancel to clear dnd-kit's activation guard.
 */

export const TOUCH_DRAG_HOLD_DELAY_MS = 400;
export const TOUCH_DRAG_MOVE_TOLERANCE_PX = 10;

/** Distance that activates a drag when no hold gate applies (mouse, pen, non-pannable surfaces). */
const DRAG_ACTIVATION_DISTANCE_PX = 6;

const ARMED_CUE_ATTRIBUTE = 'data-drag-armed';

const isScrollableOverflow = (overflow: string) =>
  overflow === 'auto' || overflow === 'scroll' || overflow === 'overlay';

/**
 * Whether the browser may pan from a touch starting on `target`: true unless a
 * `touch-action: none` appears on the element itself or an ancestor below the
 * first scroll container. (`touch-action` intersects down the ancestor chain,
 * so a `none` anywhere below the scroller forbids the pan outright.)
 */
const canSurfacePan = (target: EventTarget | null): boolean => {
  let element = target instanceof Element ? target : null;

  while (element) {
    const { overflowX, overflowY, touchAction } = getComputedStyle(element);

    if (touchAction === 'none') {
      return false;
    }

    if (isScrollableOverflow(overflowX) || isScrollableOverflow(overflowY)) {
      return true;
    }

    element = element.parentElement;
  }

  return true;
};

/** Whether a touch surface claims one-finger movement for its own gesture, so a drag needs the hold. */
const requestsTouchHold = (target: EventTarget | null): boolean =>
  target instanceof Element && target.closest('[data-drag-hold-on-touch="true"]') !== null;

const stopClickPropagation = (event: Event) => {
  event.stopPropagation();
};

export interface HoldToDragSensorOptions extends SensorOptions {
  activationConstraint?: {
    delay: number;
    tolerance: number;
  };
  onActivation?(props: { event: Event }): void;
}

type Gate = 'immediate' | 'waiting' | 'armed' | 'active';

export class HoldToDragSensor {
  static activators: Activator<HoldToDragSensorOptions>[] = [
    {
      eventName: 'onPointerDown',
      handler: (event, options) => {
        const nativeEvent = event.nativeEvent as PointerEvent;

        if (!nativeEvent.isPrimary || nativeEvent.button !== 0) {
          return false;
        }

        if (nativeEvent.pointerType !== 'touch' && nativeEvent.pointerType !== 'pen') {
          return false;
        }

        options.onActivation?.({ event: nativeEvent });
        return true;
      },
    },
  ];

  /**
   * Mirror of `TouchSensor.setup()`: a non-passive `touchmove` listener present
   * from startup makes `preventDefault()` work in touchmove listeners added
   * later (required for iOS Safari), which the active drag relies on.
   */
  static setup(): () => void {
    const noop = () => {};

    window.addEventListener('touchmove', noop, { capture: false, passive: false });

    return () => window.removeEventListener('touchmove', noop);
  }

  autoScrollEnabled = true;

  private readonly props: SensorProps<HoldToDragSensorOptions>;
  private readonly document: Document;
  private readonly pointerId: number;
  private readonly startX: number;
  private readonly startY: number;
  private readonly tolerance: number;
  private readonly cueNode: HTMLElement | null;
  private readonly abortController = new AbortController();
  private holdTimer: number | null = null;
  private gate: Gate | 'idle' = 'idle';
  private detached = false;

  constructor(props: SensorProps<HoldToDragSensorOptions>) {
    this.props = props;

    const event = props.event as PointerEvent;
    this.document = event.target instanceof Element ? event.target.ownerDocument : window.document;
    this.pointerId = event.pointerId;
    this.startX = event.clientX;
    this.startY = event.clientY;
    this.cueNode = props.activeNode.node.current;

    const { activationConstraint } = props.options ?? {};
    this.tolerance = activationConstraint?.tolerance ?? TOUCH_DRAG_MOVE_TOLERANCE_PX;

    // Pointer listeners live on the document, not the touched element: the
    // element may unmount mid-gesture (the virtualized grid under auto-scroll),
    // and document-level listeners keep receiving the events.
    const { signal } = this.abortController;
    this.document.addEventListener('pointermove', this.handlePointerMove, { signal });
    this.document.addEventListener('pointerup', this.handlePointerEnd, { signal });
    this.document.addEventListener('pointercancel', this.handlePointerCancel, { signal });
    this.document.addEventListener('pointerdown', this.handleSecondaryPointerDown, { signal });
    this.document.addEventListener('touchmove', this.handleTouchMove, { passive: false, signal });
    this.document.addEventListener('keydown', this.handleKeyDown, { signal });
    this.document.addEventListener('visibilitychange', this.handleCancel, { signal });
    window.addEventListener('resize', this.handleCancel, { signal });
    window.addEventListener('contextmenu', this.handleContextMenu, { signal });
    window.addEventListener('dragstart', this.handleNativeDragStart, { signal });

    if (event.pointerType === 'pen' || (!canSurfacePan(event.target) && !requestsTouchHold(event.target))) {
      this.gate = 'immediate';
      return;
    }

    this.gate = 'waiting';
    this.holdTimer = window.setTimeout(() => {
      if (this.gate !== 'waiting') {
        return;
      }

      this.gate = 'armed';
      this.cueNode?.setAttribute(ARMED_CUE_ATTRIBUTE, 'true');
      this.props.onPending(
        this.props.active,
        { delay: this.delay, tolerance: this.tolerance },
        { x: this.startX, y: this.startY }
      );
    }, this.delay);
  }

  private get delay(): number {
    return this.props.options?.activationConstraint?.delay ?? TOUCH_DRAG_HOLD_DELAY_MS;
  }

  private readonly handlePointerMove = (event: PointerEvent) => {
    if (event.pointerId !== this.pointerId) {
      return;
    }

    if (this.gate === 'active') {
      this.props.onMove({ x: event.clientX, y: event.clientY });
      return;
    }

    const distance = Math.hypot(event.clientX - this.startX, event.clientY - this.startY);

    if (this.gate === 'immediate') {
      if (distance > DRAG_ACTIVATION_DISTANCE_PX) {
        this.startDrag();
      }

      return;
    }

    if (this.gate === 'armed') {
      // The hold succeeded; movement beyond the tolerance is the drag itself.
      if (distance > this.tolerance) {
        this.startDrag();
      }

      return;
    }

    if (distance > this.tolerance) {
      this.endGesture(false);
    }
  };

  private readonly handlePointerEnd = (event: PointerEvent) => {
    if (event.pointerId !== this.pointerId) {
      return;
    }

    // End taps and drags on lift; only activated drags suppress the trailing click.
    this.endGesture(true);
  };

  private readonly handlePointerCancel = (event: PointerEvent) => {
    if (event.pointerId !== this.pointerId) {
      return;
    }

    // The browser claimed the gesture (a native pan). Its event stream ends
    // here, so the sensor must too — a stuck gesture blocks all later drags.
    this.cancelGesture();
  };

  private readonly handleSecondaryPointerDown = (event: PointerEvent) => {
    if (this.gate === 'active') {
      // A second finger during a live drag changes nothing.
      return;
    }

    if (!event.isPrimary) {
      // A second finger during the hold is pinch or scroll intent, never a drag.
      this.cancelGesture();
    }
  };

  private readonly handleTouchMove = (event: TouchEvent) => {
    // Prevent native touch pan after arming; pointer-event prevention cannot stop it and its threshold may precede
    // drag tolerance.
    if ((this.gate === 'armed' || this.gate === 'active') && event.cancelable) {
      event.preventDefault();
    }
  };

  private readonly handleContextMenu = (event: MouseEvent) => {
    if (this.gate === 'armed') {
      // Android's long-press menu (~500ms) outlasts the hold delay. Treat it as
      // menu intent: disarm so a drag cannot start over the open menu, and let
      // the menu through.
      this.cancelGesture();
      return;
    }

    // Mid-drag or mid-gesture: suppress the native menu, as the stock sensors do.
    event.preventDefault();
  };

  private readonly handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      this.cancelGesture();
    }
  };

  private readonly handleCancel = () => {
    this.cancelGesture();
  };

  private readonly handleNativeDragStart = (event: Event) => {
    event.preventDefault();
  };

  private startDrag() {
    this.clearHoldTimer();
    this.clearArmedCue();
    this.gate = 'active';

    // A drag that ends where it started must not also click its own tile, so
    // swallow the trailing click. Removed shortly after detach (not with the
    // other listeners) so the click that follows the lift stays swallowed.
    this.document.addEventListener('click', stopClickPropagation, true);
    this.document.getSelection()?.removeAllRanges();
    // Keep the selection clear for the whole drag, as the stock sensors do.
    this.document.addEventListener('selectionchange', this.clearSelection, { signal: this.abortController.signal });
    this.props.onStart({ x: this.startX, y: this.startY });
  }

  private readonly clearSelection = () => {
    this.document.getSelection()?.removeAllRanges();
  };

  /** Abort unactivated gestures, then always call onEnd/onCancel to clear dnd-kit's activation guard. */
  private endGesture(ended: boolean) {
    const activated = this.gate === 'active';

    this.detach();

    if (!activated) {
      this.props.onAbort(this.props.active);
    }

    if (ended) {
      this.props.onEnd();
    } else {
      this.props.onCancel();
    }
  }

  private cancelGesture() {
    this.endGesture(false);
  }

  private clearHoldTimer() {
    if (this.holdTimer !== null) {
      window.clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
  }

  private clearArmedCue() {
    this.cueNode?.removeAttribute(ARMED_CUE_ATTRIBUTE);
  }

  detach() {
    if (this.detached) {
      return;
    }

    this.detached = true;
    this.gate = 'idle';
    this.clearHoldTimer();
    this.clearArmedCue();
    this.abortController.abort();
    setTimeout(() => this.document.removeEventListener('click', stopClickPropagation, true), 50);
  }
}

/** Restrict MouseSensor to the primary button, matching the replaced PointerSensor. */
export class PrimaryMouseSensor extends MouseSensor {}

PrimaryMouseSensor.activators = [
  {
    eventName: 'onMouseDown',
    handler: ({ nativeEvent: event }, { onActivation }) => {
      if (event.button !== 0) {
        return false;
      }

      onActivation?.({ event });
      return true;
    },
  },
];
