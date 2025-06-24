import { useAppDispatch } from 'app/store/storeHooks';
import { allNonRasterLayersIsHiddenToggled } from 'features/controlLayers/store/canvasSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';

export const useCanvasToggleNonRasterLayersHotkey = () => {
  const dispatch = useAppDispatch();

  const handleToggleNonRasterLayers = useCallback(() => {
    dispatch(allNonRasterLayersIsHiddenToggled());
  }, [dispatch]);

  useRegisteredHotkeys({
    id: 'toggleNonRasterLayers',
    category: 'canvas',
    callback: handleToggleNonRasterLayers,
    dependencies: [handleToggleNonRasterLayers],
  });
}; 