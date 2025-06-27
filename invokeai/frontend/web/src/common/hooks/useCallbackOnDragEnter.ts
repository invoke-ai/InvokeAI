import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { dropTargetForExternal } from '@atlaskit/pragmatic-drag-and-drop/external/adapter';
import { useTimeoutCallback } from 'common/hooks/useTimeoutCallback';
import type { RefObject } from 'react';
import { useEffect } from 'react';

export const useCallbackOnDragEnter = (cb: () => void, ref: RefObject<HTMLElement>, delay = 300) => {
  const [run, cancel] = useTimeoutCallback(cb, delay);

  useEffect(() => {
    const element = ref.current;
    if (!element) {
      return;
    }
    return combine(
      dropTargetForElements({
        element,
        onDragEnter: run,
        onDragLeave: cancel,
      }),
      dropTargetForExternal({
        element,
        onDragEnter: run,
        onDragLeave: cancel,
      })
    );
  }, [cancel, ref, run]);
};
