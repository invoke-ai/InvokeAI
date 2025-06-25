import { useAppStore } from 'app/store/nanostores/store';
import { debounce } from 'es-toolkit/compat';
import type { Dimensions } from 'features/controlLayers/store/types';
import { selectUiSlice, textAreaSizesStateChanged } from 'features/ui/store/uiSlice';
import { type RefObject, useCallback, useEffect, useMemo } from 'react';

type Options = {
  trackWidth: boolean;
  trackHeight: boolean;
  initialWidth?: number;
  initialHeight?: number;
};

/**
 * Persists the width and/or height of a text area to redux.
 * @param id The unique id of this textarea, used as key to storage
 * @param ref A ref to the textarea element
 * @param options.trackWidth Whether to track width
 * @param options.trackHeight Whether to track width
 * @param options.initialWidth An optional initial width in pixels
 * @param options.initialHeight An optional initial height in pixels
 */
export const usePersistedTextAreaSize = (id: string, ref: RefObject<HTMLTextAreaElement>, options: Options) => {
  const { dispatch, getState } = useAppStore();

  const onResize = useCallback(
    (size: Partial<Dimensions>) => {
      dispatch(textAreaSizesStateChanged({ id, size }));
    },
    [dispatch, id]
  );

  const debouncedOnResize = useMemo(() => debounce(onResize, 300), [onResize]);

  useEffect(() => {
    const el = ref.current;
    if (!el) {
      return;
    }

    // Nothing to do here if we are not tracking anything.
    if (!options.trackHeight && !options.trackWidth) {
      return;
    }

    // Before registering the observer, grab the stored size from state - we may need to restore the size.
    const storedSize = selectUiSlice(getState()).textAreaSizes[id];

    // Prefer to restore the stored size, falling back to initial size if it exists
    if (storedSize?.width !== undefined) {
      el.style.width = `${storedSize.width}px`;
    } else if (options.initialWidth !== undefined) {
      el.style.width = `${options.initialWidth}px`;
    }

    if (storedSize?.height !== undefined) {
      el.style.height = `${storedSize.height}px`;
    } else if (options.initialHeight !== undefined) {
      el.style.height = `${options.initialHeight}px`;
    }

    let currentHeight = el.offsetHeight;
    let currentWidth = el.offsetWidth;

    const resizeObserver = new ResizeObserver(() => {
      // We only want to push the changes if a tracked dimension changes
      let didChange = false;
      const newSize: Partial<Dimensions> = {};

      if (options.trackHeight) {
        if (el.offsetHeight !== currentHeight) {
          didChange = true;
          currentHeight = el.offsetHeight;
        }
        newSize.height = currentHeight;
      }

      if (options.trackWidth) {
        if (el.offsetWidth !== currentWidth) {
          didChange = true;
          currentWidth = el.offsetWidth;
        }
        newSize.width = currentWidth;
      }

      if (didChange) {
        debouncedOnResize(newSize);
      }
    });

    resizeObserver.observe(el);

    return () => {
      debouncedOnResize.cancel();
      resizeObserver.disconnect();
    };
  }, [
    debouncedOnResize,
    dispatch,
    getState,
    id,
    options.initialHeight,
    options.initialWidth,
    options.trackHeight,
    options.trackWidth,
    ref,
  ]);
};
