import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { selectIsStaging } from 'features/controlLayers/store/canvasSessionSlice';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isStaging = useAppSelector(selectIsStaging);

  const deleteSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null) {
      return;
    }
    dispatch(entityDeleted({ entityIdentifier: selectedEntityIdentifier }));
  }, [dispatch, selectedEntityIdentifier]);

  const isDeleteEnabled = useMemo(
    () => selectedEntityIdentifier !== null && !isStaging,
    [selectedEntityIdentifier, isStaging]
  );

  useHotkeys(['delete', 'backspace'], deleteSelectedLayer, { enabled: isDeleteEnabled }, [
    isDeleteEnabled,
    deleteSelectedLayer,
  ]);
}
