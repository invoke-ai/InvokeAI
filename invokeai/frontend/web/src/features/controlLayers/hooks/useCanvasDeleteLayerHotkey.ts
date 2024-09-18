import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { $canvasRightPanelTab } from 'features/controlLayers/components/CanvasRightPanel';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const canvasRightPanelTab = useStore($canvasRightPanelTab);
  const appTab = useAppSelector(selectActiveTab);

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

  useHotkeys(['delete', 'backspace'], deleteSelectedLayer, { enabled: isDeleteEnabled }, [
    isDeleteEnabled,
    deleteSelectedLayer,
  ]);
}
