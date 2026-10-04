import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsLocked } from 'features/controlLayers/hooks/useEntityIsLocked';
import { entityReset } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isMaskEntityIdentifier } from 'features/controlLayers/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo } from 'react';

export function useCanvasResetLayerHotkey() {
  useAssertSingleton(useCanvasResetLayerHotkey.name);
  const dispatch = useAppDispatch();
  const entityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const adapter = useEntityAdapterSafe(entityIdentifier);
  const isLocked = useEntityIsLocked(entityIdentifier);

  const resetSelectedLayer = useCallback(() => {
    if (entityIdentifier === null || adapter === null) {
      return;
    }
    adapter.bufferRenderer.clearBuffer();
    dispatch(entityReset({ entityIdentifier }));
  }, [adapter, dispatch, entityIdentifier]);

  const isResetAllowed = useMemo(
    () => entityIdentifier !== null && isMaskEntityIdentifier(entityIdentifier),
    [entityIdentifier]
  );

  useRegisteredHotkeys({
    id: 'resetSelected',
    category: 'canvas',
    callback: resetSelectedLayer,
    options: { enabled: isResetAllowed && !isBusy && !isLocked },
    dependencies: [isResetAllowed, isBusy, isLocked, resetSelectedLayer],
  });
}
