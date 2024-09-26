import type { RefObject } from 'react';
import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type {
  ImperativePanelGroupHandle,
  ImperativePanelHandle,
  PanelOnCollapse,
  PanelOnExpand,
  PanelProps,
  PanelResizeHandleProps,
} from 'react-resizable-panels';
import { getPanelGroupElement, getResizeHandleElementsForGroup } from 'react-resizable-panels';

type Direction = 'horizontal' | 'vertical';

const NO_SIZE = Symbol('NO_SIZE');

export type UsePanelOptions = {
  id: string;
  /**
   * The minimum size of the panel in pixels. Must be at least 1.
   */
  minSizePx: number;
  /**
   * The default size of the panel in pixels. Must be at least 1.
   */
  defaultSizePx?: number;
  /**
   * The direction of the panel group.
   * This is required to accurately calculate the available space for the panel, minus the space taken by the handles.
   */
  panelGroupDirection: Direction;
  /**
   * A ref to the panel group.
   */
  imperativePanelGroupRef: RefObject<ImperativePanelGroupHandle>;
  /**
   * Called when the board's collapsed state changes.
   */
  onCollapse?: (isCollapsed: boolean) => void;
};

export type UsePanelReturn = {
  /**
   * Whether the panel is collapsed.
   */
  isCollapsed: boolean;
  /**
   * Reset the panel to the minSize.
   */
  reset: () => void;
  /**
   * Toggle the panel between collapsed and expanded.
   */
  toggle: () => void;
  /**
   * Expand the panel.
   */
  expand: () => void;
  /**
   * Collapse the panel.
   */
  collapse: () => void;
  /**
   * Resize the panel to the given size in pixels. Must be at least 1.
   */
  resize: (sizePx: number) => void;
  /**
   * The props to apply to the panel.
   */
  panelProps: Partial<PanelProps & { ref: RefObject<ImperativePanelHandle> }>;
  /**
   * The props to apply to the resize handle.
   */
  resizeHandleProps: Partial<PanelResizeHandleProps>;
};

export const usePanel = (arg: UsePanelOptions): UsePanelReturn => {
  const imperativePanelRef = useRef<ImperativePanelHandle>(null);
  const [_minSize, _setMinSize] = useState(1);
  const [_defaultSize, _setDefaultSize] = useState(1);

  const observerCallback = useCallback(() => {
    if (!imperativePanelRef?.current) {
      return;
    }

    const minSizePct = getSizeAsPercentage(arg.minSizePx, arg.imperativePanelGroupRef, arg.panelGroupDirection);
    const defaultSizePct = getSizeAsPercentage(
      arg.defaultSizePx ?? arg.minSizePx,
      arg.imperativePanelGroupRef,
      arg.panelGroupDirection
    );

    if (minSizePct === NO_SIZE || defaultSizePct === NO_SIZE) {
      // This can happen when the panel is hidden
      return;
    }

    _setMinSize(minSizePct);

    if (defaultSizePct > minSizePct) {
      _setDefaultSize(defaultSizePct);
    } else {
      _setDefaultSize(minSizePct);
    }

    const currentSize = imperativePanelRef.current.getSize();
    const isCollapsed = imperativePanelRef.current.isCollapsed();

    if (!isCollapsed && currentSize < minSizePct && minSizePct > 0) {
      imperativePanelRef.current.resize(minSizePct);
    }
  }, [arg.defaultSizePx, arg.imperativePanelGroupRef, arg.minSizePx, arg.panelGroupDirection]);

  // If the units are pixels, we need to calculate the min size as a percentage of the available space,
  // then resize the panel if it is too small.
  useLayoutEffect(() => {
    if (!arg.imperativePanelGroupRef.current) {
      return;
    }
    const id = arg.imperativePanelGroupRef.current.getId();
    const panelGroupElement = getPanelGroupElement(id);
    const panelGroupHandleElements = getResizeHandleElementsForGroup(id);
    if (!panelGroupElement) {
      return;
    }
    const resizeObserver = new ResizeObserver(observerCallback);

    resizeObserver.observe(panelGroupElement);
    panelGroupHandleElements.forEach((el) => resizeObserver.observe(el));

    observerCallback();

    return () => {
      resizeObserver.disconnect();
    };
  }, [arg.imperativePanelGroupRef, observerCallback]);

  const [isCollapsed, setIsCollapsed] = useState(() => Boolean(imperativePanelRef.current?.isCollapsed()));

  const onCollapse = useCallback<PanelOnCollapse>(() => {
    setIsCollapsed(true);
    arg.onCollapse?.(true);
  }, [arg]);

  const onExpand = useCallback<PanelOnExpand>(() => {
    setIsCollapsed(false);
    arg.onCollapse?.(false);
  }, [arg]);

  const toggle = useCallback(() => {
    if (imperativePanelRef.current?.isCollapsed()) {
      imperativePanelRef.current?.expand();
    } else {
      imperativePanelRef.current?.collapse();
    }
  }, []);

  const expand = useCallback(() => {
    imperativePanelRef.current?.expand();
  }, []);

  const collapse = useCallback(() => {
    imperativePanelRef.current?.collapse();
  }, []);

  const resize = useCallback(
    (sizePx: number) => {
      // We need to calculate the size as a percentage of the available space
      const sizeAsPct = getSizeAsPercentage(sizePx, arg.imperativePanelGroupRef, arg.panelGroupDirection);
      if (sizeAsPct === NO_SIZE) {
        return;
      }
      imperativePanelRef.current?.resize(sizeAsPct);
    },
    [arg]
  );

  const reset = useCallback(() => {
    imperativePanelRef.current?.resize(_minSize);
  }, [_minSize]);

  const cycleState = useCallback(() => {
    // If the panel is really super close to the min size, collapse it
    if (Math.abs((imperativePanelRef.current?.getSize() ?? 0) - _defaultSize) < 0.01) {
      collapse();
      return;
    }

    // Otherwise, resize to the min size
    imperativePanelRef.current?.resize(_defaultSize);
  }, [_defaultSize, collapse]);

  return {
    isCollapsed,
    reset,
    toggle,
    expand,
    collapse,
    resize,
    panelProps: {
      id: arg.id,
      defaultSize: _defaultSize,
      onCollapse,
      onExpand,
      ref: imperativePanelRef,
      minSize: _minSize,
    },
    resizeHandleProps: {
      onDoubleClick: cycleState,
    },
  };
};

/**
 * For a desired size in pixels, calculates the size of the panel as a percentage of the available space.
 * @param sizeInPixels The desired size of the panel in pixels.
 * @param panelGroupHandleRef The ref to the panel group handle.
 * @param panelGroupDirection The direction of the panel group.
 * @returns The size of the panel as a percentage.
 */
const getSizeAsPercentage = (
  sizeInPixels: number,
  panelGroupHandleRef: RefObject<ImperativePanelGroupHandle>,
  panelGroupDirection: Direction
) => {
  if (!panelGroupHandleRef.current) {
    // No panel group handle ref, so we can't calculate the size
    return NO_SIZE;
  }
  const id = panelGroupHandleRef.current.getId();
  const panelGroupElement = getPanelGroupElement(id);
  if (!panelGroupElement) {
    // No panel group element, size is 0
    return NO_SIZE;
  }

  // The available space is the width/height of the panel group...
  let availableSpace =
    panelGroupDirection === 'horizontal' ? panelGroupElement.offsetWidth : panelGroupElement.offsetHeight;

  if (!availableSpace) {
    // No available space, size is 0
    return NO_SIZE;
  }

  // ...minus the width/height of the resize handles
  getResizeHandleElementsForGroup(id).forEach((el) => {
    availableSpace -= panelGroupDirection === 'horizontal' ? el.offsetWidth : el.offsetHeight;
  });

  // The final value is a percentage of the available space - must be between 1 and 100, inclusive
  return Math.max(Math.min((sizeInPixels / availableSpace) * 100, 100), 1);
};
