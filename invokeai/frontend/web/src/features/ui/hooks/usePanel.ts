import { useCallback, useLayoutEffect, useRef, useState } from 'react';
import type {
  ImperativePanelHandle,
  PanelOnCollapse,
  PanelOnExpand,
} from 'react-resizable-panels';

export type UsePanelOptions =
  | { minSize: number; unit: 'percentages' }
  | {
      minSize: number;
      unit: 'pixels';
      fallbackMinSizePct: number;
      panelGroupID: string;
    };

export const usePanel = (arg: UsePanelOptions) => {
  const panelHandleRef = useRef<ImperativePanelHandle>(null);
  const newMinSizeRef = useRef<number>(0);
  const currentSizeRef = useRef<number>(0);
  const [_minSize, _setMinSize] = useState<number>(
    arg.unit === 'percentages' ? arg.minSize : arg.fallbackMinSizePct
  );

  useLayoutEffect(() => {
    if (arg.unit === 'percentages') {
      return;
    }
    const panelGroupElement = document.querySelector(
      `[data-panel-group][data-panel-group-id="${arg.panelGroupID}"]`
    );
    if (!panelGroupElement) {
      return;
    }
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (!panelHandleRef?.current) {
          return;
        }
        newMinSizeRef.current = (arg.minSize * 100) / entry.contentRect.width;
        currentSizeRef.current =
          panelHandleRef.current.getSize() ?? arg.fallbackMinSizePct;
        if (currentSizeRef.current < newMinSizeRef.current) {
          panelHandleRef.current.resize(newMinSizeRef.current);
        }
        _setMinSize(newMinSizeRef.current);
      }
    });

    resizeObserver.observe(panelGroupElement);
    // _setMinSize(
    //   (arg.minSize * 100) / panelGroupElement.getBoundingClientRect().width
    // );

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
