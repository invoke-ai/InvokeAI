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

export type UsePanelOptions = {
  id: string;
  /**
   * The minimum size of the panel in pixels.
   */
  minSize: number;
  /**
   * The default size of the panel in pixels.
   */
  defaultSize?: number;
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
   * Resize the panel to the given size in the same units as the minSize.
   */
  resize: (size: number) => void;
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
  const [_minSize, _setMinSize] = useState<number>(0);
  const [_defaultSize, _setDefaultSize] = useState<number>(0);

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
    const resizeObserver = new ResizeObserver(() => {
      if (!imperativePanelRef?.current) {
        return;
      }

      const minSizePct = getSizeAsPercentage(arg.minSize, arg.imperativePanelGroupRef, arg.panelGroupDirection);
      const defaultSizePct = getSizeAsPercentage(
        arg.defaultSize || arg.minSize,
        arg.imperativePanelGroupRef,
        arg.panelGroupDirection
      );

      if (minSizePct > 100) {
        // This can happen when the panel is hidden
        return;
      }

      _setMinSize(minSizePct);

      if (defaultSizePct && defaultSizePct > minSizePct) {
        _setDefaultSize(defaultSizePct);
      } else {
        _setDefaultSize(minSizePct);
      }

      const currentSize = imperativePanelRef.current.getSize();
      const isCollapsed = imperativePanelRef.current.isCollapsed();

      if (isCollapsed) {
        return;
      }

      if (!isCollapsed && currentSize < minSizePct && minSizePct > 0) {
        imperativePanelRef.current.resize(minSizePct);
      }
    });

    resizeObserver.observe(panelGroupElement);
    panelGroupHandleElements.forEach((el) => resizeObserver.observe(el));

    if (imperativePanelRef.current) {
      const currentSize = imperativePanelRef.current.getSize();
      const isCollapsed = imperativePanelRef.current.isCollapsed();

      // Resize the panel to the min size once on startup if it is too small
      if (!isCollapsed && currentSize < _minSize) {
        imperativePanelRef.current.resize(_minSize);
      }
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [_minSize, arg]);

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
    (size: number) => {
      // We need to calculate the size as a percentage of the available space
      const sizeAsPct = getSizeAsPercentage(size, arg.imperativePanelGroupRef, arg.panelGroupDirection);
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
    return 0;
  }
  const id = panelGroupHandleRef.current.getId();
  const panelGroupElement = getPanelGroupElement(id);
  if (!panelGroupElement) {
    // No panel group element, size is 0
    return 0;
  }

  // The available space is the width/height of the panel group...
  let availableSpace =
    panelGroupDirection === 'horizontal' ? panelGroupElement.offsetWidth : panelGroupElement.offsetHeight;

  // ...minus the width/height of the resize handles
  getResizeHandleElementsForGroup(id).forEach((el) => {
    availableSpace -= panelGroupDirection === 'horizontal' ? el.offsetWidth : el.offsetHeight;
  });

  // The final value is a percentage of the available space
  return (sizeInPixels / availableSpace) * 100;
};
