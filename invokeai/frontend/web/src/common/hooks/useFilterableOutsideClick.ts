/* eslint-disable @typescript-eslint/no-explicit-any */

/**
 * Adapted from https://github.com/chakra-ui/chakra-ui/blob/v2/packages/hooks/src/use-outside-click.ts
 *
 * The main change here is to support filtering of outside clicks via a `filter` function.
 *
 * This lets us work around issues with portals and components like popovers, which typically close on an outside click.
 *
 * For example, consider a popover that has a custom drop-down component inside it, which uses a portal to render
 * the drop-down options. The original outside click handler would close the popover when clicking on the drop-down options,
 * because the click is outside the popover - but we expect the popover to stay open in this case.
 *
 * A filter function like this can fix that:
 *
 * ```ts
 * const filter = (el: HTMLElement) => el.className.includes('chakra-portal') || el.id.includes('react-select')
 * ```
 *
 * This ignores clicks on react-select-based drop-downs and Chakra UI portals and is used as the default filter.
 */

import { useCallback, useEffect, useRef } from 'react';

type FilterFunction = (el: HTMLElement | SVGElement) => boolean;

export function useCallbackRef<T extends (...args: any[]) => any>(
  callback: T | undefined,
  deps: React.DependencyList = []
) {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  });

  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useCallback(((...args) => callbackRef.current?.(...args)) as T, deps);
}

export interface UseOutsideClickProps {
  /**
   * Whether the hook is enabled
   */
  enabled?: boolean;
  /**
   * The reference to a DOM element.
   */
  ref: React.RefObject<HTMLElement | null>;
  /**
   * Function invoked when a click is triggered outside the referenced element.
   */
  handler?: (e: Event) => void;
  /**
   * A function that filters the elements that should be considered as outside clicks.
   *
   * If omitted, a default filter function that ignores clicks in Chakra UI portals and react-select components is used.
   */
  filter?: FilterFunction;
}

export const DEFAULT_FILTER: FilterFunction = (el) => {
  if (el instanceof SVGElement) {
    // SVGElement's type appears to be incorrect. Its className is not a string, which causes `includes` to fail.
    // Let's assume that SVG elements with a class name are not part of the portal and should not be filtered.
    return false;
  }
  return el.className.includes('chakra-portal') || el.id.includes('react-select');
};

/**
 * Example, used in components like Dialogs and Popovers, so they can close
 * when a user clicks outside them.
 */
export function useFilterableOutsideClick(props: UseOutsideClickProps) {
  const { ref, handler, enabled = true, filter = DEFAULT_FILTER } = props;
  const savedHandler = useCallbackRef(handler);

  const stateRef = useRef({
    isPointerDown: false,
    ignoreEmulatedMouseEvents: false,
  });

  const state = stateRef.current;

  useEffect(() => {
    if (!enabled) {
      return;
    }
    const onPointerDown: any = (e: PointerEvent) => {
      if (isValidEvent(e, ref, filter)) {
        state.isPointerDown = true;
      }
    };

    const onMouseUp: any = (event: MouseEvent) => {
      if (state.ignoreEmulatedMouseEvents) {
        state.ignoreEmulatedMouseEvents = false;
        return;
      }

      if (state.isPointerDown && handler && isValidEvent(event, ref)) {
        state.isPointerDown = false;
        savedHandler(event);
      }
    };

    const onTouchEnd = (event: TouchEvent) => {
      state.ignoreEmulatedMouseEvents = true;
      if (handler && state.isPointerDown && isValidEvent(event, ref)) {
        state.isPointerDown = false;
        savedHandler(event);
      }
    };

    const doc = getOwnerDocument(ref.current);
    doc.addEventListener('mousedown', onPointerDown, true);
    doc.addEventListener('mouseup', onMouseUp, true);
    doc.addEventListener('touchstart', onPointerDown, true);
    doc.addEventListener('touchend', onTouchEnd, true);

    return () => {
      doc.removeEventListener('mousedown', onPointerDown, true);
      doc.removeEventListener('mouseup', onMouseUp, true);
      doc.removeEventListener('touchstart', onPointerDown, true);
      doc.removeEventListener('touchend', onTouchEnd, true);
    };
  }, [handler, ref, savedHandler, state, enabled, filter]);
}

function isValidEvent(event: Event, ref: React.RefObject<HTMLElement | null>, filter?: FilterFunction): boolean {
  const target = (event.composedPath?.()[0] ?? event.target) as HTMLElement;

  if (target) {
    const doc = getOwnerDocument(target);
    if (!doc.contains(target)) {
      return false;
    }
  }

  if (ref.current?.contains(target)) {
    return false;
  }

  // This is the main logic change from the original hook.
  if (filter) {
    // Check if the click is inside an element matching the filter.
    // This is used for portal-awareness or other general exclusion cases.
    let currentElement: HTMLElement | null = target;
    // Traverse up the DOM tree from the target element.
    while (currentElement && currentElement !== document.body) {
      if (filter(currentElement)) {
        return false;
      }
      currentElement = currentElement.parentElement;
    }
  }

  // If the click is not inside the ref and not inside a portal, it's a valid outside click.
  return true;
}

function getOwnerDocument(node?: Element | null): Document {
  return node?.ownerDocument ?? document;
}
