import { useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { ImperativePanelHandle, MixedSizes } from 'react-resizable-panels';

export const usePanel = (minSize: Partial<MixedSizes>) => {
  const ref = useRef<ImperativePanelHandle>(null);

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
    reset,
    toggle,
    expand,
    collapse,
  };
};
