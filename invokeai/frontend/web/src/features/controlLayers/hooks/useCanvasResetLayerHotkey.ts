import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import {
  imReset,
  layerReset,
  rgReset,
  selectCanvasV2Slice,
} from 'features/controlLayers/store/canvasV2Slice';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const selectSelectedEntityIdentifier = createMemoizedSelector(
  selectCanvasV2Slice,
  (canvasV2State) => canvasV2State.selectedEntityIdentifier
);

export function useCanvasResetLayerHotkey() {
  useAssertSingleton(useCanvasResetLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);

  const resetSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null) {
      return;
    }
    const { type, id } = selectedEntityIdentifier;
    if (type === 'layer') {
      dispatch(layerReset({ id }));
    }
    if (type === 'regional_guidance') {
      dispatch(rgReset({ id }));
    }
    if (type === 'inpaint_mask') {
      dispatch(imReset());
    }
  }, [dispatch, selectedEntityIdentifier]);

  const isResetEnabled = useMemo(
    () =>
      (!isStaging && selectedEntityIdentifier?.type === 'layer') ||
      selectedEntityIdentifier?.type === 'regional_guidance' ||
      selectedEntityIdentifier?.type === 'inpaint_mask',
    [isStaging, selectedEntityIdentifier?.type]
  );

  useHotkeys('shift+c', resetSelectedLayer, { enabled: isResetEnabled }, [
    isResetEnabled,
    isStaging,
    resetSelectedLayer,
  ]);
}
