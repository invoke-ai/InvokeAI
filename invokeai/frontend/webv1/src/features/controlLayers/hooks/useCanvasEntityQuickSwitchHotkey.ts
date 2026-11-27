import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import {
  selectBookmarkedEntityIdentifier,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useEffect, useRef } from 'react';

type SelectionHistory = {
  prev: CanvasEntityIdentifier | null;
  current: CanvasEntityIdentifier | null;
};

export const useCanvasEntityQuickSwitchHotkey = () => {
  const dispatch = useAppDispatch();
  const selectionHistoryRef = useRef<SelectionHistory>({ prev: null, current: null });
  const selected = useAppSelector(selectSelectedEntityIdentifier);
  const bookmarked = useAppSelector(selectBookmarkedEntityIdentifier);

  useEffect(() => {
    const { current } = selectionHistoryRef.current;

    if (current?.id !== selected?.id) {
      selectionHistoryRef.current = { prev: current, current: selected };
    }
  }, [selected]);

  const onQuickSwitch = useCallback(() => {
    const { prev, current } = selectionHistoryRef.current;

    if (bookmarked) {
      if (current?.id !== bookmarked.id) {
        selectionHistoryRef.current = { prev: current, current: bookmarked };
        dispatch(entitySelected({ entityIdentifier: bookmarked }));
      } else if (prev) {
        selectionHistoryRef.current = { prev, current: prev };
        dispatch(entitySelected({ entityIdentifier: prev }));
      }
    } else if (prev !== null && current !== null) {
      selectionHistoryRef.current = { prev: current, current: prev };
      dispatch(entitySelected({ entityIdentifier: prev }));
    }
  }, [bookmarked, dispatch]);

  useRegisteredHotkeys({
    id: 'quickSwitch',
    category: 'canvas',
    callback: onQuickSwitch,
    dependencies: [onQuickSwitch],
  });
};
