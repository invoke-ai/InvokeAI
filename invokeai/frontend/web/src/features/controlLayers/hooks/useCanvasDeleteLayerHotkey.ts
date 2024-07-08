import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import {
  caDeleted,
  ipaDeleted,
  layerDeleted,
  rgDeleted,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const selectSelectedEntityIdentifier = createMemoizedSelector(
  selectCanvasV2Slice,
  (canvasV2State) => canvasV2State.selectedEntityIdentifier
);

export function useCanvasDeleteLayerHotkey() {
  useAssertSingleton(useCanvasDeleteLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);

  const deleteSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null) {
      return;
    }
    const { type, id } = selectedEntityIdentifier;
    if (type === 'layer') {
      dispatch(layerDeleted({ id }));
    }
    if (type === 'regional_guidance') {
      dispatch(rgDeleted({ id }));
    }
    if (type === 'control_adapter') {
      dispatch(caDeleted({ id }));
    }
    if (type === 'ip_adapter') {
      dispatch(ipaDeleted({ id }));
    }
  }, [dispatch, selectedEntityIdentifier]);

  const isDeleteEnabled = useMemo(
    () => selectedEntityIdentifier !== null && !isStaging,
    [selectedEntityIdentifier, isStaging]
  );

  useHotkeys('shift+d', deleteSelectedLayer, { enabled: isDeleteEnabled }, [isDeleteEnabled, deleteSelectedLayer]);
}
