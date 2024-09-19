import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import {
  selectBookmarkedEntityIdentifier,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useEffect, useState } from 'react';

export const useCanvasEntityQuickSwitchHotkey = () => {
  const dispatch = useAppDispatch();
  const [prev, setPrev] = useState<CanvasEntityIdentifier | null>(null);
  const [current, setCurrent] = useState<CanvasEntityIdentifier | null>(null);
  const selected = useAppSelector(selectSelectedEntityIdentifier);
  const bookmarked = useAppSelector(selectBookmarkedEntityIdentifier);
  const imageViewer = useImageViewer();

  // Update prev and current when selected entity changes
  useEffect(() => {
    if (current?.id !== selected?.id) {
      setPrev(current);
      setCurrent(selected);
    }
  }, [current, selected]);

  const onQuickSwitch = useCallback(() => {
    if (bookmarked) {
      if (current?.id !== bookmarked.id) {
        // Switch between current (non-bookmarked) and bookmarked
        setPrev(current);
        setCurrent(bookmarked);
        dispatch(entitySelected({ entityIdentifier: bookmarked }));
      } else if (prev) {
        // Switch back to the last non-bookmarked entity
        setCurrent(prev);
        dispatch(entitySelected({ entityIdentifier: prev }));
      }
    } else if (prev !== null && current !== null) {
      // Switch between prev and current if no bookmarked entity
      setPrev(current);
      setCurrent(prev);
      dispatch(entitySelected({ entityIdentifier: prev }));
    }
  }, [bookmarked, current, dispatch, prev]);

  useRegisteredHotkeys({
    id: 'quickSwitch',
    category: 'canvas',
    callback: onQuickSwitch,
    options: { enabled: !imageViewer.isOpen },
    dependencies: [onQuickSwitch, imageViewer.isOpen],
  });
};
