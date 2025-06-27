import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTabCanvasRightPanel } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const canvasRightPanelTab = useAppSelector(selectActiveTabCanvasRightPanel);

  const deleteSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null || isBusy || canvasRightPanelTab !== 'layers') {
      return;
    }
    dispatch(entityDeleted({ entityIdentifier: selectedEntityIdentifier }));
  }, [canvasRightPanelTab, dispatch, isBusy, selectedEntityIdentifier]);

  useRegisteredHotkeys({
    id: 'deleteSelected',
    category: 'canvas',
    callback: deleteSelectedLayer,
    dependencies: [deleteSelectedLayer],
  });
}
