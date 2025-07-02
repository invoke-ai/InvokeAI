import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { inpaintMaskInverted } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectSelectedEntityIdentifier } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback, useMemo } from 'react';

export const useCanvasInvertMaskHotkey = () => {
  useAssertSingleton('useCanvasInvertMaskHotkey');
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const canvasSlice = useAppSelector(selectCanvasSlice);
  const isBusy = useCanvasIsBusy();

  const handleInvertMask = useCallback(() => {
    if (!selectedEntityIdentifier || selectedEntityIdentifier.type !== 'inpaint_mask') {
      return;
    }

    // Check if the selected entity has objects and there's a valid bounding box
    const entity = canvasSlice.inpaintMasks.entities.find((entity) => entity.id === selectedEntityIdentifier.id);
    const hasObjects = entity?.objects && entity.objects.length > 0;
    const hasBbox = canvasSlice.bbox.rect.width > 0 && canvasSlice.bbox.rect.height > 0;

    if (!hasObjects || !hasBbox) {
      return;
    }

    dispatch(
      inpaintMaskInverted({ entityIdentifier: selectedEntityIdentifier as CanvasEntityIdentifier<'inpaint_mask'> })
    );
  }, [dispatch, selectedEntityIdentifier, canvasSlice]);

  const isInvertMaskAllowed = useMemo(() => {
    if (!selectedEntityIdentifier || selectedEntityIdentifier.type !== 'inpaint_mask') {
      return false;
    }

    const entity = canvasSlice.inpaintMasks.entities.find((entity) => entity.id === selectedEntityIdentifier.id);
    const hasObjects = entity?.objects && entity.objects.length > 0;
    const hasBbox = canvasSlice.bbox.rect.width > 0 && canvasSlice.bbox.rect.height > 0;

    return hasObjects && hasBbox;
  }, [selectedEntityIdentifier, canvasSlice]);

  useRegisteredHotkeys({
    id: 'invertMask',
    category: 'canvas',
    callback: handleInvertMask,
    options: { enabled: isInvertMaskAllowed && !isBusy, preventDefault: true },
    dependencies: [isInvertMaskAllowed, isBusy, handleInvertMask],
  });
};
