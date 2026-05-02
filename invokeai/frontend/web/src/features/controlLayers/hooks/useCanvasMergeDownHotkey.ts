import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { getIsCanvasMergeDownHotkeyEnabled } from 'features/controlLayers/hooks/canvasMergeHotkeyUtils';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIdentifierBelowThisOne } from 'features/controlLayers/hooks/useNextRenderableEntityIdentifier';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo, useRef } from 'react';

export const useCanvasMergeDownHotkey = () => {
  useAssertSingleton(useCanvasMergeDownHotkey.name);
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const entityIdentifierBelowThisOne = useEntityIdentifierBelowThisOne(selectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const isMergeInFlightRef = useRef(false);

  const mergeDown = useCallback(() => {
    if (!selectedEntityIdentifier || !entityIdentifierBelowThisOne || isMergeInFlightRef.current) {
      return;
    }
    isMergeInFlightRef.current = true;
    void canvasManager.compositor
      .mergeByEntityIdentifiers([entityIdentifierBelowThisOne, selectedEntityIdentifier], true)
      .finally(() => {
        isMergeInFlightRef.current = false;
      });
  }, [canvasManager.compositor, entityIdentifierBelowThisOne, selectedEntityIdentifier]);

  const isEnabled = useMemo(
    () => getIsCanvasMergeDownHotkeyEnabled(selectedEntityIdentifier, entityIdentifierBelowThisOne, isBusy),
    [entityIdentifierBelowThisOne, isBusy, selectedEntityIdentifier]
  );

  useRegisteredHotkeys({
    id: 'mergeDown',
    category: 'canvas',
    callback: mergeDown,
    options: { enabled: isEnabled, preventDefault: true },
    dependencies: [isEnabled, mergeDown],
  });
};
