import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { selectActiveTab, selectActiveTabCanvasRightPanel } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const canvasRightPanelTab = useAppSelector(selectActiveTabCanvasRightPanel);
  const appTab = useAppSelector(selectActiveTab);

  const imageViewer = useImageViewer();

  const deleteSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null) {
      return;
    }
    dispatch(entityDeleted({ entityIdentifier: selectedEntityIdentifier }));
  }, [dispatch, selectedEntityIdentifier]);

  const isDeleteEnabled = useMemo(
    () => selectedEntityIdentifier !== null && !isBusy && canvasRightPanelTab === 'layers' && appTab === 'canvas',
    [selectedEntityIdentifier, isBusy, canvasRightPanelTab, appTab]
  );

  useRegisteredHotkeys({
    id: 'deleteSelected',
    category: 'canvas',
    callback: deleteSelectedLayer,
    options: { enabled: isDeleteEnabled && !imageViewer.isOpen },
    dependencies: [isDeleteEnabled, deleteSelectedLayer, imageViewer.isOpen],
  });
}
