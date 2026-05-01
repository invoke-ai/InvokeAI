import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useVisibleEntityCountByType } from 'features/controlLayers/hooks/useVisibleEntityCountByType';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo } from 'react';

export const useCanvasMergeVisibleHotkey = () => {
  useAssertSingleton(useCanvasMergeVisibleHotkey.name);
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const visibleEntityCount = useVisibleEntityCountByType(selectedEntityIdentifier?.type ?? 'raster_layer');

  const mergeVisible = useCallback(() => {
    if (!selectedEntityIdentifier) {
      return;
    }
    canvasManager.compositor.mergeVisibleOfType(selectedEntityIdentifier.type);
  }, [canvasManager.compositor, selectedEntityIdentifier]);

  const isEnabled = useMemo(() => {
    if (!selectedEntityIdentifier) {
      return false;
    }
    if (visibleEntityCount <= 1) {
      return false;
    }
    if (isBusy) {
      return false;
    }
    return true;
  }, [isBusy, selectedEntityIdentifier, visibleEntityCount]);

  useRegisteredHotkeys({
    id: 'mergeVisible',
    category: 'canvas',
    callback: mergeVisible,
    options: { enabled: isEnabled, preventDefault: true },
    dependencies: [isEnabled, mergeVisible],
  });
};
