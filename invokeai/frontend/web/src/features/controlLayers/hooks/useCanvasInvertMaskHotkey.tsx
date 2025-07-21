import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useInvertMask } from 'features/controlLayers/hooks/useInvertMask';
import { selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import { isInpaintMaskEntityIdentifier } from 'features/controlLayers/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useMemo } from 'react';

export const useCanvasInvertMaskHotkey = () => {
  useAssertSingleton('useCanvasInvertMaskHotkey');
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const invertMask = useInvertMask();

  const isEnabled = useMemo(() => {
    if (!selectedEntityIdentifier) {
      return false;
    }
    if (!isInpaintMaskEntityIdentifier(selectedEntityIdentifier)) {
      return false;
    }
    if (isBusy) {
      return false;
    }
    return true;
  }, [selectedEntityIdentifier, isBusy]);

  useRegisteredHotkeys({
    id: 'invertMask',
    category: 'canvas',
    callback: invertMask,
    options: { enabled: isEnabled, preventDefault: true },
    dependencies: [invertMask, isEnabled],
  });
};
