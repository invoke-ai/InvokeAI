import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent, RefObject } from 'react';

import { chakra, useSlotRecipe } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { resizeHandleSlotRecipe } from '@theme/recipes';
import { useCallback, useRef } from 'react';

export type PointerDragEnd = 'release' | 'interrupt' | 'escape';

interface PointerDragOptions {
  /** Shown everywhere for the whole gesture, so a pointer outrunning the handle does not flicker. */
  cursor: string;
  /** Called at most once per animation frame with the offset from the gesture's start. */
  onMove: (deltaX: number, deltaY: number) => void;
  onEnd: (reason: PointerDragEnd) => void;
}

/**
 * Run one pointer gesture from a pointerdown. Moves are coalesced to animation frames; Escape ends it as `escape`,
 * `pointercancel` as `interrupt`. Aborting the returned controller ends it without calling `onEnd`.
 */
export const beginPointerDrag = (
  event: ReactPointerEvent<HTMLElement>,
  { cursor, onEnd, onMove }: PointerDragOptions
): AbortController => {
  event.preventDefault();

  const { clientX: startX, clientY: startY, pointerId } = event;
  const root = document.documentElement;
  const session = new AbortController();
  const { signal } = session;
  let lastX = startX;
  let lastY = startY;
  let frame = 0;

  try {
    event.currentTarget.setPointerCapture(pointerId);
  } catch {
    // The window listeners carry the gesture without capture.
  }

  root.style.setProperty('--wb-pointer-drag-cursor', cursor);
  root.setAttribute('data-pointer-drag', '');
  signal.addEventListener('abort', () => {
    cancelAnimationFrame(frame);
    root.removeAttribute('data-pointer-drag');
    root.style.removeProperty('--wb-pointer-drag-cursor');
  });

  const flush = () => {
    frame = 0;
    onMove(lastX - startX, lastY - startY);
  };
  const finish = (reason: PointerDragEnd) => {
    // Apply a move still waiting on its frame, and cancel that frame so nothing runs after the end.
    if (frame !== 0) {
      cancelAnimationFrame(frame);
      if (reason !== 'escape') {
        flush();
      }
      frame = 0;
    }
    try {
      onEnd(reason);
    } finally {
      session.abort();
    }
  };

  window.addEventListener(
    'pointermove',
    (moveEvent) => {
      if (moveEvent.pointerId !== pointerId) {
        return;
      }
      // Another application can swallow the pointerup.
      if (moveEvent.buttons === 0) {
        finish('release');
        return;
      }
      lastX = moveEvent.clientX;
      lastY = moveEvent.clientY;
      if (frame === 0) {
        frame = requestAnimationFrame(flush);
      }
    },
    { signal }
  );
  window.addEventListener('pointerup', (upEvent) => upEvent.pointerId === pointerId && finish('release'), { signal });
  window.addEventListener(
    'pointercancel',
    (cancelEvent) => cancelEvent.pointerId === pointerId && finish('interrupt'),
    {
      signal,
    }
  );
  window.addEventListener(
    'keydown',
    (keyEvent) => {
      if (keyEvent.key === 'Escape') {
        keyEvent.preventDefault();
        keyEvent.stopPropagation();
        finish('escape');
      }
    },
    { capture: true, signal }
  );

  return session;
};

/** Owns one gesture at a time and ends it silently when the component unmounts. */
export const usePointerDrag = (): ((
  event: ReactPointerEvent<HTMLElement>,
  options: PointerDragOptions
) => AbortSignal) => {
  const sessionRef = useRef<AbortController | null>(null);

  useMountEffect(() => () => sessionRef.current?.abort());

  return useCallback((event, options) => {
    sessionRef.current?.abort();
    const session = beginPointerDrag(event, options);
    sessionRef.current = session;
    return session.signal;
  }, []);
};

let activeResizeCount = 0;
const resizeListeners = new Set<() => void>();

const setResizeActive = (isActive: boolean) => {
  activeResizeCount += isActive ? 1 : -1;
  for (const listener of resizeListeners) {
    listener();
  }
};

/** True while a resize handle is being dragged; costly layout consumers may wait for it to end. */
export const isResizeDragActive = (): boolean => activeResizeCount > 0;

export const subscribeResizeDrag = (listener: () => void): (() => void) => {
  resizeListeners.add(listener);
  return () => resizeListeners.delete(listener);
};

/** Pointer travel past the collapse point before a drag disarms it again, so the boundary does not flicker. */
const COLLAPSE_HYSTERESIS_PX = 40;
const DEFAULT_STEP = { '%': 2, px: 16 } as const;

export interface ResizeCollapse {
  /** Raw drag value at or below which releasing collapses the pane. */
  at: number;
  /** Size the pane previews while a release would collapse it. */
  preview: number;
  onCollapse: (source: 'keyboard' | 'pointer') => void;
}

export interface ResizeHandleProps {
  collapse?: ResizeCollapse;
  label: string;
  /** Hides the divider line while something else, such as a region outline, is drawn over it. */
  lineHidden?: boolean;
  max: number;
  min: number;
  /** The divider line's direction: a vertical divider resizes widths. */
  orientation: 'horizontal' | 'vertical';
  /** Which side of the divider the sized pane is on. */
  pane: 'after' | 'before';
  /** Receives the live size inline while dragging, so the drag renders without React. */
  paneRef: RefObject<HTMLElement | null>;
  /** Size when layout squeezes the pane below `value`; drives ARIA and the keyboard collapse floor. */
  renderedValue?: number;
  sizeProperty?: 'flexBasis' | 'height' | 'width';
  step?: number;
  unit?: '%' | 'px';
  value: number;
  onCommit: (value: number) => void;
}

/**
 * The shared divider for resizable panes: a hairline with a centered grip, keyboard stepping, and an optional
 * drag-past-the-floor collapse. Callers render the committed size; drags write the live size to `paneRef`.
 */
export const ResizeHandle = ({
  collapse,
  label,
  lineHidden = false,
  max,
  min,
  orientation,
  pane,
  paneRef,
  renderedValue: renderedValueProp,
  sizeProperty: sizePropertyProp,
  step: stepProp,
  unit = 'px',
  value,
  onCommit,
}: ResizeHandleProps) => {
  const renderedValue = renderedValueProp ?? value;
  const sizeProperty = sizePropertyProp ?? (orientation === 'vertical' ? 'width' : 'height');
  const step = stepProp ?? DEFAULT_STEP[unit];
  const styles = useSlotRecipe({ recipe: resizeHandleSlotRecipe })({ orientation });
  const startDrag = usePointerDrag();
  const handleRef = useRef<HTMLDivElement>(null);
  const isVertical = orientation === 'vertical';
  const growSign = pane === 'before' ? 1 : -1;
  const clamp = useCallback((next: number) => Math.min(max, Math.max(min, next)), [max, min]);

  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const paneElement = paneRef.current;
      const handle = handleRef.current;
      if (event.button !== 0 || !paneElement || !handle) {
        return;
      }

      const container = paneElement.parentElement;
      const containerPx = container ? (isVertical ? container.clientWidth : container.clientHeight) : 0;
      const unitsPerPx = unit === '%' ? (containerPx > 0 ? 100 / containerPx : 0) : 1;
      let next = clamp(value);
      let isArmed = false;
      let hasEnded = false;

      const write = (size: number | null) => {
        if (size === null) {
          paneElement.style.removeProperty(toCssProperty(sizeProperty));
          return;
        }
        paneElement.style.setProperty(toCssProperty(sizeProperty), `${size}${unit}`);
      };
      const setArmed = (armed: boolean) => {
        isArmed = armed;
        paneElement.toggleAttribute('data-collapse-armed', armed);
        handle.toggleAttribute('data-collapse-armed', armed);
      };

      handle.setAttribute('data-dragging', '');
      setResizeActive(true);
      const signal = startDrag(event, {
        cursor: isVertical ? 'col-resize' : 'row-resize',
        onEnd: (reason) => {
          const shouldCollapse = isArmed && reason === 'release';
          if (reason !== 'escape') {
            if (shouldCollapse) {
              collapse?.onCollapse('pointer');
            } else if (next !== value) {
              onCommit(next);
            }
          }
          // Set after the commit: if it throws, the abort clears the live size instead.
          hasEnded = true;
          const clear = () => {
            write(null);
            setArmed(false);
          };
          if (reason === 'escape') {
            clear();
          } else {
            // Hold the live size one more frame so the committed render replaces it without a flash.
            requestAnimationFrame(clear);
          }
        },
        onMove: (deltaX, deltaY) => {
          const raw = value + (isVertical ? deltaX : deltaY) * growSign * unitsPerPx;
          if (collapse) {
            setArmed(raw <= collapse.at + (isArmed ? COLLAPSE_HYSTERESIS_PX * unitsPerPx : 0));
          }
          next = clamp(unit === 'px' ? Math.round(raw) : Math.round(raw * 10) / 10);
          write(isArmed && collapse ? collapse.preview : next);
        },
      });
      signal.addEventListener('abort', () => {
        handle.removeAttribute('data-dragging');
        setResizeActive(false);
        if (!hasEnded) {
          write(null);
          setArmed(false);
        }
      });
    },
    [clamp, collapse, growSign, isVertical, onCommit, paneRef, sizeProperty, startDrag, unit, value]
  );

  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const stepBy = event.shiftKey ? step * 2 : step;
      const [grow, shrink] = isVertical
        ? growSign === 1
          ? ['ArrowRight', 'ArrowLeft']
          : ['ArrowLeft', 'ArrowRight']
        : growSign === 1
          ? ['ArrowDown', 'ArrowUp']
          : ['ArrowUp', 'ArrowDown'];
      const change =
        event.key === grow
          ? stepBy
          : event.key === shrink
            ? -stepBy
            : event.key === 'End'
              ? max - value
              : event.key === 'Home'
                ? min - value
                : undefined;
      if (change === undefined) {
        return;
      }
      event.preventDefault();
      // A further shrink at the floor collapses.
      if (collapse && change < 0 && renderedValue <= min) {
        collapse.onCollapse('keyboard');
        return;
      }
      const next = clamp(value + change);
      if (next !== value) {
        onCommit(next);
      }
    },
    [clamp, collapse, growSign, isVertical, max, min, onCommit, renderedValue, step, value]
  );

  return (
    <chakra.div css={styles.root} data-line-hidden={lineHidden || undefined}>
      <chakra.div
        ref={handleRef}
        aria-label={label}
        aria-orientation={orientation}
        aria-valuemax={max}
        aria-valuemin={Math.min(min, renderedValue)}
        aria-valuenow={renderedValue}
        css={styles.handle}
        role="separator"
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
      />
    </chakra.div>
  );
};

const toCssProperty = (property: NonNullable<ResizeHandleProps['sizeProperty']>) =>
  property === 'flexBasis' ? 'flex-basis' : property;

/** A window's bottom-right resize grip; the caller applies the 2D offset. */
export const ResizeCorner = ({
  label,
  valueMin,
  valueNow,
  onDragCancel,
  onDragEnd,
  onDragMove,
  onKeyDown,
}: {
  label: string;
  valueMin: number;
  valueNow: number;
  /** The gesture stopped without ending, e.g. the corner unmounted, or `onDragEnd` threw; undo any live preview. */
  onDragCancel: () => void;
  onDragEnd: (reason: PointerDragEnd) => void;
  onDragMove: (deltaX: number, deltaY: number) => void;
  onKeyDown: (event: ReactKeyboardEvent<HTMLDivElement>) => void;
}) => {
  const styles = useSlotRecipe({ recipe: resizeHandleSlotRecipe })({ orientation: 'corner' });
  const startDrag = usePointerDrag();
  const onPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }
      const handle = event.currentTarget;
      let hasEnded = false;
      handle.setAttribute('data-dragging', '');
      setResizeActive(true);
      const onEnd = (reason: PointerDragEnd) => {
        onDragEnd(reason);
        hasEnded = true;
      };
      startDrag(event, { cursor: 'nwse-resize', onEnd, onMove: onDragMove }).addEventListener('abort', () => {
        handle.removeAttribute('data-dragging');
        setResizeActive(false);
        if (!hasEnded) {
          onDragCancel();
        }
      });
    },
    [onDragCancel, onDragEnd, onDragMove, startDrag]
  );

  return (
    <chakra.div
      aria-label={label}
      aria-valuemin={valueMin}
      aria-valuenow={valueNow}
      css={styles.handle}
      role="separator"
      tabIndex={0}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
    />
  );
};
