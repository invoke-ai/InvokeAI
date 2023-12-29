import { useCallback, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import type {
  ImperativePanelHandle,
  PanelOnCollapse,
  PanelOnExpand,
} from 'react-resizable-panels';

export const usePanel = (minSize: number) => {
  const ref = useRef<ImperativePanelHandle>(null);

  const [isCollapsed, setIsCollapsed] = useState(() =>
    Boolean(ref.current?.isCollapsed())
  );

  const onCollapse = useCallback<PanelOnCollapse>(() => {
    setIsCollapsed(true);
  }, []);

  const onExpand = useCallback<PanelOnExpand>(() => {
    setIsCollapsed(false);
  }, []);

  const toggle = useCallback(() => {
    if (ref.current?.isCollapsed()) {
      flushSync(() => {
        ref.current?.expand();
      });
    } else {
      flushSync(() => {
        ref.current?.collapse();
      });
    }
  }, []);

  const expand = useCallback(() => {
    flushSync(() => {
      ref.current?.expand();
    });
  }, []);

  const collapse = useCallback(() => {
    flushSync(() => {
      ref.current?.collapse();
    });
  }, []);

  const reset = useCallback(() => {
    flushSync(() => {
      ref.current?.resize(minSize);
    });
  }, [minSize]);

  return {
    ref,
    minSize,
    isCollapsed,
    onCollapse,
    onExpand,
    reset,
    toggle,
    expand,
    collapse,
  };
};
