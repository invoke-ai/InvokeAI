import type { RefObject } from 'react';
import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type {
  ImperativePanelGroupHandle,
  ImperativePanelHandle,
  PanelOnCollapse,
  PanelOnExpand,
} from 'react-resizable-panels';
import { getPanelGroupElement, getResizeHandleElementsForGroup } from 'react-resizable-panels';

type Direction = 'horizontal' | 'vertical';

export type UsePanelOptions =
  | {
      /**
       * The minimum size of the panel as a percentage.
       */
      minSize: number;
      /**
       * The unit of the minSize
       */
      unit: 'percentages';
    }
  | {
      /**
       * The minimum size of the panel in pixels.
       */
      minSize: number;
      /**
       * The unit of the minSize.
       */
      unit: 'pixels';
      /**
       * The direction of the panel group.
       * This is required to accurately calculate the available space for the panel, minus the space taken by the handles.
       */
      panelGroupDirection: Direction;
      /**
       * A ref to the panel group.
       */
      panelGroupRef: RefObject<ImperativePanelGroupHandle>;
    };

export type UsePanelReturn = {
  /**
   * The ref to the panel handle.
   */
  ref: RefObject<ImperativePanelHandle>;
  /**
   * The dynamically calculated minimum size of the panel.
   */
  minSize: number;
  /**
   * Whether the panel is collapsed.
   */
  isCollapsed: boolean;
  /**
   * The onCollapse callback. This is required to update the isCollapsed state.
   * This should be passed to the panel as the onCollapse prop. Wrap it if additional logic is required.
   */
  onCollapse: PanelOnCollapse;
  /**
   * The onExpand callback. This is required to update the isCollapsed state.
   * This should be passed to the panel as the onExpand prop. Wrap it if additional logic is required.
   */
  onExpand: PanelOnExpand;
  /**
   * Reset the panel to the minSize.
   */
  reset: () => void;
  /**
   * Reset the panel to the minSize. If the panel is already at the minSize, collapse it.
   * This should be passed to the `onDoubleClick` prop of the panel's nearest resize handle.
   */
  onDoubleClickHandle: () => void;
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
};

export const usePanel = (arg: UsePanelOptions): UsePanelReturn => {
  const panelHandleRef = useRef<ImperativePanelHandle>(null);
  const [_minSize, _setMinSize] = useState<number>(arg.unit === 'percentages' ? arg.minSize : 0);

  // If the units are pixels, we need to calculate the min size as a percentage of the available space,
  // then resize the panel if it is too small.
  useLayoutEffect(() => {
    if (arg.unit === 'percentages' || !arg.panelGroupRef.current) {
      return;
    }
    const id = arg.panelGroupRef.current.getId();
    const panelGroupElement = getPanelGroupElement(id);
    const panelGroupHandleElements = getResizeHandleElementsForGroup(id);
    if (!panelGroupElement) {
      return;
    }
    const resizeObserver = new ResizeObserver(() => {
      if (!panelHandleRef?.current) {
        return;
      }

      const minSizePct = getSizeAsPercentage(arg.minSize, arg.panelGroupRef, arg.panelGroupDirection);

      _setMinSize(minSizePct);

      /**
       * TODO(psyche): Ideally, we only resize the panel if there is not enough room for it in the
       * panel group. This is a bit tricky, though. We'd need to track the last known panel size
       * and compare it to the new size before resizing. This introduces some complexity that I'd
       * rather not need to maintain.
       *
       * For now, we'll just resize the panel to the min size every time the panel group is resized.
       */
      if (!panelHandleRef.current.isCollapsed()) {
        panelHandleRef.current.resize(minSizePct);
      }
    });

    resizeObserver.observe(panelGroupElement);
    panelGroupHandleElements.forEach((el) => resizeObserver.observe(el));

    // Resize the panel to the min size once on startup
    const minSizePct = getSizeAsPercentage(arg.minSize, arg.panelGroupRef, arg.panelGroupDirection);
    panelHandleRef.current?.resize(minSizePct);

    return () => {
      resizeObserver.disconnect();
    };
  }, [arg]);

  const [isCollapsed, setIsCollapsed] = useState(() => Boolean(panelHandleRef.current?.isCollapsed()));

  const onCollapse = useCallback<PanelOnCollapse>(() => {
    setIsCollapsed(true);
  }, []);

  const onExpand = useCallback<PanelOnExpand>(() => {
    setIsCollapsed(false);
  }, []);

  const toggle = useCallback(() => {
    if (panelHandleRef.current?.isCollapsed()) {
      panelHandleRef.current?.expand();
    } else {
      panelHandleRef.current?.collapse();
    }
  }, []);

  const expand = useCallback(() => {
    panelHandleRef.current?.expand();
  }, []);

  const collapse = useCallback(() => {
    panelHandleRef.current?.collapse();
  }, []);

  const resize = useCallback(
    (size: number) => {
      // If we are using percentages, we can just resize to the given size
      if (arg.unit === 'percentages') {
        panelHandleRef.current?.resize(size);
        return;
      }

      // If we are using pixels, we need to calculate the size as a percentage of the available space
      const sizeAsPct = getSizeAsPercentage(size, arg.panelGroupRef, arg.panelGroupDirection);
      panelHandleRef.current?.resize(sizeAsPct);
    },
    [arg]
  );

  const reset = useCallback(() => {
    panelHandleRef.current?.resize(_minSize);
  }, [_minSize]);

  const onDoubleClickHandle = useCallback(() => {
    // If the panel is really super close to the min size, collapse it
    if (Math.abs((panelHandleRef.current?.getSize() ?? 0) - _minSize) < 0.01) {
      collapse();
      return;
    }

    // Otherwise, resize to the min size
    panelHandleRef.current?.resize(_minSize);
  }, [_minSize, collapse]);

  return {
    ref: panelHandleRef,
    minSize: _minSize,
    isCollapsed,
    onCollapse,
    onExpand,
    reset,
    toggle,
    expand,
    collapse,
    resize,
    onDoubleClickHandle,
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
