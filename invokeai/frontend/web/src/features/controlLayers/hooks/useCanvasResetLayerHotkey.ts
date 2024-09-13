import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useEntityAdapterSafe } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityReset } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { isMaskEntityIdentifier } from 'features/controlLayers/store/types';
import type { ReadableAtom } from 'nanostores';
import { atom } from 'nanostores';
import { useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const selectSelectedEntityIdentifier = createMemoizedSelector(
  selectCanvasSlice,
  (canvasState) => canvasState.selectedEntityIdentifier
);

const $fallbackFalse: ReadableAtom<boolean> = atom(false);

export function useCanvasResetLayerHotkey() {
  useAssertSingleton(useCanvasResetLayerHotkey.name);
  const dispatch = useAppDispatch();
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const isBusy = useCanvasIsBusy();
  const adapter = useEntityAdapterSafe(selectedEntityIdentifier);
  const isInteractable = useStore(adapter?.$isInteractable ?? $fallbackFalse);

  const resetSelectedLayer = useCallback(() => {
    if (selectedEntityIdentifier === null) {
      return;
    }
    dispatch(entityReset({ entityIdentifier: selectedEntityIdentifier }));
  }, [dispatch, selectedEntityIdentifier]);

  const isResetEnabled = useMemo(
    () => selectedEntityIdentifier !== null && isMaskEntityIdentifier(selectedEntityIdentifier),
    [selectedEntityIdentifier]
  );

  useHotkeys('shift+c', resetSelectedLayer, { enabled: isResetEnabled && !isBusy && isInteractable }, [
    isResetEnabled,
    isBusy,
    isInteractable,
    resetSelectedLayer,
  ]);
}
