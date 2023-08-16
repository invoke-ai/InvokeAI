// adapted from https://github.com/bvaughn/react-resizable-panels/issues/141#issuecomment-1540048714

import {
  RefObject,
  useCallback,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { ImperativePanelHandle } from 'react-resizable-panels';

export const useMinimumPanelSize = (
  minSizePx: number,
  defaultSizePct: number,
  groupId: string,
  orientation: 'horizontal' | 'vertical' = 'horizontal'
): {
  ref: RefObject<ImperativePanelHandle>;
  minSizePct: number;
} => {
  const ref = useRef<ImperativePanelHandle>(null);
  const [minSizePct, setMinSizePct] = useState(defaultSizePct);

  const handleWindowResize = useCallback(() => {
    const size = ref.current?.getSize();

    if (size !== undefined && size < minSizePct) {
      ref.current?.resize(minSizePct);
    }
  }, [minSizePct]);

  useLayoutEffect(() => {
    const panelGroup = document.querySelector(
      `[data-panel-group-id="${groupId}"]`
    );
    const resizeHandles = document.querySelectorAll(
      orientation === 'horizontal'
        ? '.resize-handle-horizontal'
        : '.resize-handle-vertical'
    );

    if (!panelGroup) {
      return;
    }
    const observer = new ResizeObserver(() => {
      let dim =
        orientation === 'horizontal'
          ? panelGroup.getBoundingClientRect().width
          : panelGroup.getBoundingClientRect().height;

      resizeHandles.forEach((resizeHandle) => {
        dim -=
          orientation === 'horizontal'
            ? resizeHandle.getBoundingClientRect().width
            : resizeHandle.getBoundingClientRect().height;
      });

      // Minimum size in pixels is a percentage of the PanelGroup's width/height
      setMinSizePct((minSizePx / dim) * 100);
    });
    observer.observe(panelGroup);
    resizeHandles.forEach((resizeHandle) => {
      observer.observe(resizeHandle);
    });

    window.addEventListener('resize', handleWindowResize);

    return () => {
      observer.disconnect();
      window.removeEventListener('resize', handleWindowResize);
    };
  }, [groupId, handleWindowResize, minSizePct, minSizePx, orientation]);

  return { ref, minSizePct };
};
