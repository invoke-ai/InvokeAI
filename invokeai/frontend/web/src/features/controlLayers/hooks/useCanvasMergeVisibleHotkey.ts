import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { getIsCanvasMergeVisibleHotkeyEnabled } from 'features/controlLayers/hooks/canvasMergeHotkeyUtils';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useVisibleEntityCountByType } from 'features/controlLayers/hooks/useVisibleEntityCountByType';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo, useRef } from 'react';

export const useCanvasMergeVisibleHotkey = () => {
  useAssertSingleton(useCanvasMergeVisibleHotkey.name);
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const visibleEntityCount = useVisibleEntityCountByType(selectedEntityIdentifier?.type ?? 'raster_layer');
  const isMergeInFlightRef = useRef(false);

  const mergeVisible = useCallback(() => {
    if (!selectedEntityIdentifier || isMergeInFlightRef.current) {
      return;
    }
    isMergeInFlightRef.current = true;
    void canvasManager.compositor.mergeVisibleOfType(selectedEntityIdentifier.type).finally(() => {
      isMergeInFlightRef.current = false;
    });
  }, [canvasManager.compositor, selectedEntityIdentifier]);

  const isEnabled = useMemo(
    () => getIsCanvasMergeVisibleHotkeyEnabled(selectedEntityIdentifier, visibleEntityCount, isBusy),
    [isBusy, selectedEntityIdentifier, visibleEntityCount]
  );

  useRegisteredHotkeys({
    id: 'mergeVisible',
    category: 'canvas',
    callback: mergeVisible,
    options: { enabled: isEnabled, preventDefault: true },
    dependencies: [isEnabled, mergeVisible],
  });
};
