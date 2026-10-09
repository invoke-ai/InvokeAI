import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getFocusedRegion } from 'common/hooks/focus';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';

export const getCanvasDeleteTarget = (
  focusedRegion: ReturnType<typeof getFocusedRegion>,
  isBusy: boolean,
  isEditing: boolean
): 'path' | 'layer' | null => {
  if (isBusy || (focusedRegion !== 'canvas' && focusedRegion !== 'layers')) {
    return null;
  }
  if (isEditing) {
    return 'path';
  }
  return focusedRegion === 'layers' ? 'layer' : null;
};

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();

  const deleteSelected = useCallback(() => {
    const pathTool = canvasManager.tool.tools.path;
    const target = getCanvasDeleteTarget(getFocusedRegion(), isBusy, pathTool.hasActiveEditSession());
    if (target === 'path') {
      pathTool.deleteSelectedPointsOrActivePath();
      return;
    }

    if (target !== 'layer' || selectedEntityIdentifier === null) {
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
