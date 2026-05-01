import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIdentifierBelowThisOne } from 'features/controlLayers/hooks/useNextRenderableEntityIdentifier';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo } from 'react';

export const useCanvasMergeDownHotkey = () => {
  useAssertSingleton(useCanvasMergeDownHotkey.name);
  const canvasManager = useCanvasManager();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const entityIdentifierBelowThisOne = useEntityIdentifierBelowThisOne(selectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();

  const mergeDown = useCallback(() => {
    if (!selectedEntityIdentifier || !entityIdentifierBelowThisOne) {
      return;
    }
    canvasManager.compositor.mergeByEntityIdentifiers([entityIdentifierBelowThisOne, selectedEntityIdentifier], true);
  }, [canvasManager.compositor, entityIdentifierBelowThisOne, selectedEntityIdentifier]);

  const isEnabled = useMemo(() => {
    if (!selectedEntityIdentifier || !entityIdentifierBelowThisOne) {
      return false;
    }
    if (isBusy) {
      return false;
    }
    return true;
  }, [entityIdentifierBelowThisOne, isBusy, selectedEntityIdentifier]);

  useRegisteredHotkeys({
    id: 'mergeDown',
    category: 'canvas',
    callback: mergeDown,
    options: { enabled: isEnabled, preventDefault: true },
    dependencies: [isEnabled, mergeDown],
  });
};
