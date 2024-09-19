import { useStore } from '@nanostores/react';
import { $false } from 'app/store/nanostores/util';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityReset } from 'features/controlLayers/store/canvasSlice';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isMaskEntityIdentifier } from 'features/controlLayers/store/types';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo } from 'react';

export function useCanvasResetLayerHotkey() {
  useAssertSingleton(useCanvasResetLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const adapter = useEntityAdapterSafe(selectedEntityIdentifier);
  const isInteractable = useStore(adapter?.$isInteractable ?? $false);
  const imageViewer = useImageViewer();

  const resetSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null || adapter === null) {
      return;
    }
    adapter.bufferRenderer.clearBuffer();
    dispatch(entityReset({ entityIdentifier: selectedEntityIdentifier }));
  }, [adapter, dispatch, selectedEntityIdentifier]);

  const isResetEnabled = useMemo(
    () => selectedEntityIdentifier !== null && isMaskEntityIdentifier(selectedEntityIdentifier),
    [selectedEntityIdentifier]
  );

  useRegisteredHotkeys({
    id: 'resetSelected',
    category: 'canvas',
    callback: resetSelectedLayer,
    options: { enabled: isResetEnabled && !isBusy && isInteractable && !imageViewer.isOpen },
    dependencies: [isResetEnabled, isBusy, isInteractable, resetSelectedLayer, imageViewer.isOpen],
  });
}
