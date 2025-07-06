import { useAppDispatch } from 'app/store/storeHooks';
import { settingsSnapToGridToggled } from 'features/controlLayers/store/canvasSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';

export const useCanvasSnapToGridHotkey = () => {
  const dispatch = useAppDispatch();

  const handleToggleSnapToGrid = useCallback(() => {
    dispatch(settingsSnapToGridToggled());
  }, [dispatch]);

  useRegisteredHotkeys({
    id: 'snapToGrid',
    category: 'canvas',
    callback: handleToggleSnapToGrid,
    dependencies: [handleToggleSnapToGrid],
  });
};
