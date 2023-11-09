import { useCallback, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { ImperativePanelHandle, Units } from 'react-resizable-panels';

export const usePanel = (minSize: number, units: Units) => {
  const ref = useRef<ImperativePanelHandle>(null);

  const [isCollapsed, setIsCollapsed] = useState(() =>
    Boolean(ref.current?.getCollapsed())
  );

  const toggle = useCallback(() => {
    if (ref.current?.getCollapsed()) {
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
      ref.current?.resize(minSize, units);
    });
  }, [minSize, units]);

  return {
    ref,
    minSize,
    isCollapsed,
    setIsCollapsed,
    reset,
    toggle,
    expand,
    collapse,
  };
};
