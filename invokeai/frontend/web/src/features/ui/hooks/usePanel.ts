import type { RefObject } from 'react';
import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type {
  ImperativePanelGroupHandle,
  ImperativePanelHandle,
  PanelOnCollapse,
  PanelOnExpand,
} from 'react-resizable-panels';
import {
  getPanelGroupElement,
  getResizeHandleElementsForGroup,
} from 'react-resizable-panels';

export type UsePanelOptions =
  | { minSize: number; unit: 'percentages' }
  | {
      minSize: number;
      unit: 'pixels';
      fallbackMinSizePct: number;
      panelGroupRef: RefObject<ImperativePanelGroupHandle>;
      panelGroupDirection: 'horizontal' | 'vertical';
    };

export const usePanel = (arg: UsePanelOptions) => {
  const panelHandleRef = useRef<ImperativePanelHandle>(null);
  const [_minSize, _setMinSize] = useState<number>(
    arg.unit === 'percentages' ? arg.minSize : arg.fallbackMinSizePct
  );

  // If the units are pixels, we need to calculate the min size as a percentage of the available space
  useLayoutEffect(() => {
    if (arg.unit === 'percentages' || !arg.panelGroupRef.current) {
      return;
    }
    const panelGroupElement = getPanelGroupElement(
      arg.panelGroupRef.current.getId()
    );
    const panelGroupHandleElements = getResizeHandleElementsForGroup(
      arg.panelGroupRef.current.getId()
    );
    if (!panelGroupElement) {
      return;
    }
    const resizeObserver = new ResizeObserver(() => {
      if (!panelHandleRef?.current) {
        return;
      }
      // Calculate the available space for the panel, minus the space taken by the handles
      let dim =
        arg.panelGroupDirection === 'horizontal'
          ? panelGroupElement.offsetWidth
          : panelGroupElement.offsetHeight;

      panelGroupHandleElements.forEach(
        (el) =>
          (dim -=
            arg.panelGroupDirection === 'horizontal'
              ? el.offsetWidth
              : el.offsetHeight)
      );

      // Calculate the min size as a percentage of the available space
      const minSize = (arg.minSize * 100) / dim;
      // Must store this to avoid race conditions
      const currentSize = panelHandleRef.current.getSize();
      // Resize if the current size is smaller than the new min size - happens when the window is resized smaller
      if (currentSize < minSize) {
        panelHandleRef.current.resize(minSize);
      }

      _setMinSize(minSize);
    });

    resizeObserver.observe(panelGroupElement);

    return () => {
      resizeObserver.disconnect();
    };
  }, [arg]);

  const [isCollapsed, setIsCollapsed] = useState(() =>
    Boolean(panelHandleRef.current?.isCollapsed())
  );

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

  const reset = useCallback(() => {
    // If the panel is really super close to the min size, collapse it
    const shouldCollapse =
      Math.abs((panelHandleRef.current?.getSize() ?? 0) - _minSize) < 0.01;
    if (shouldCollapse) {
      collapse();
    } else {
      panelHandleRef.current?.resize(_minSize);
    }
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
  };
};
