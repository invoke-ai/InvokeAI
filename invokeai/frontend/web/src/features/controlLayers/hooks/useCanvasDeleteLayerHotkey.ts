import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getFocusedRegion } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();

  const deleteSelected = useCallback(() => {
    if (isBusy) {
      return;
    }

    const pathTool = canvasManager.tool.tools.path;
    if (pathTool.hasActiveEditSession()) {
      pathTool.deleteActivePath();
      return;
    }

    if (selectedEntityIdentifier === null || getFocusedRegion() !== 'layers') {
      return;
    }

    dispatch(entityDeleted({ entityIdentifier: selectedEntityIdentifier }));
  }, [canvasManager.tool.tools.path, dispatch, isBusy, selectedEntityIdentifier]);

  useRegisteredHotkeys({
    id: 'deleteSelected',
    category: 'canvas',
    callback: deleteSelected,
    dependencies: [deleteSelected],
  });
}
