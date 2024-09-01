import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import {
  selectCanvasSlice,
  selectEntity,
  selectQuickSwitchEntityIdentifier,
  selectSelectedEntityIdentifier,
} from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { atom } from 'nanostores';
import { useCallback, useEffect, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const $selectedEntityBuffer = atom<CanvasEntityIdentifier | null>(null);

export const useCanvasEntityQuickSwitchHotkey = () => {
  useAssertSingleton('useCanvasEntityQuickSwitch');

  const dispatch = useAppDispatch();
  const selectedEntityBuffer = useStore($selectedEntityBuffer);
  const quickSwitchEntityIdentifier = useAppSelector(selectQuickSwitchEntityIdentifier);
  const selectedEntityIdentifier = useAppSelector(selectSelectedEntityIdentifier);
  const selectBufferEntityIdentifier = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) =>
        selectedEntityBuffer ? (selectEntity(canvas, selectedEntityBuffer) ?? null) : null
      ),
    [selectedEntityBuffer]
  );
  const bufferEntityIdentifier = useAppSelector(selectBufferEntityIdentifier);

  const quickSwitch = useCallback(() => {
    if (quickSwitchEntityIdentifier !== null) {
      // If there is a quick switch entity, we should switch between it and the buffer
      if (quickSwitchEntityIdentifier.id !== selectedEntityIdentifier?.id) {
        // The quick switch entity is not selected - select it
        dispatch(entitySelected({ entityIdentifier: quickSwitchEntityIdentifier }));
        $selectedEntityBuffer.set(selectedEntityIdentifier);
      } else if (bufferEntityIdentifier !== null) {
        // The quick switch entity is already selected - select the buffer
        dispatch(entitySelected({ entityIdentifier: bufferEntityIdentifier }));
        $selectedEntityBuffer.set(quickSwitchEntityIdentifier);
      }
    } else {
      // No quick switch entity, so we should switch between buffer and selected entity
      // If there is no selected entity or buffer, we should not do anything
      if (selectedEntityIdentifier === null && bufferEntityIdentifier === null) {
        return;
      }
      // If there is no selected entity but we do have a buffer, we should select the buffer
      if (selectedEntityIdentifier === null && bufferEntityIdentifier !== null) {
        dispatch(entitySelected({ entityIdentifier: bufferEntityIdentifier }));
        return;
      }
      // If there is a selected entity but no buffer, we should buffer the selected entity
      if (selectedEntityIdentifier !== null && bufferEntityIdentifier === null) {
        $selectedEntityBuffer.set(selectedEntityIdentifier);
        return;
      }
      // If there is a selected entity and a buffer, and they are different, we should swap the selected entity and the buffer
      if (
        selectedEntityIdentifier !== null &&
        bufferEntityIdentifier !== null &&
        selectedEntityIdentifier.id !== bufferEntityIdentifier.id
      ) {
        $selectedEntityBuffer.set(selectedEntityIdentifier);
        dispatch(entitySelected({ entityIdentifier: bufferEntityIdentifier }));
        return;
      }
    }
  }, [bufferEntityIdentifier, dispatch, quickSwitchEntityIdentifier, selectedEntityIdentifier]);

  useEffect(() => {
    if (!bufferEntityIdentifier) {
      $selectedEntityBuffer.set(null);
    }
  }, [bufferEntityIdentifier]);

  useHotkeys('q', quickSwitch, { enabled: true, preventDefault: true }, [quickSwitch]);
};
