import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getFocusedRegion } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();

  const deleteSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null || isBusy || getFocusedRegion() !== 'layers') {
      return;
    }
    dispatch(entityDeleted({ entityIdentifier: selectedEntityIdentifier }));
  }, [dispatch, isBusy, selectedEntityIdentifier]);

  useRegisteredHotkeys({
    id: 'deleteSelected',
    category: 'canvas',
    callback: deleteSelectedLayer,
    dependencies: [deleteSelectedLayer],
  });
}
