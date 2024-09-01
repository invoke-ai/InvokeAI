import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import {
  selectCanvasSlice,
  selectEntity,
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
  const selectedEntity = useAppSelector(selectSelectedEntityIdentifier);
  const selectDoesBufferExist = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        if (!selectedEntityBuffer) {
          return true;
        }
        const bufferEntity = selectEntity(canvas, selectedEntityBuffer);
        if (bufferEntity) {
          return true;
        }
        return false;
      }),
    [selectedEntityBuffer]
  );
  const doesBufferExist = useAppSelector(selectDoesBufferExist);

  const quickSwitch = useCallback(() => {
    // If there is no selected entity or buffer, we should not do anything
    if (selectedEntity === null && selectedEntityBuffer === null) {
      return;
    }
    // If there is no selected entity but we do have a buffer, we should select the buffer
    if (selectedEntity === null && selectedEntityBuffer !== null) {
      dispatch(entitySelected({ entityIdentifier: selectedEntityBuffer }));
      return;
    }
    // If there is a selected entity but no buffer, we should buffer the selected entity
    if (selectedEntity !== null && selectedEntityBuffer === null) {
      $selectedEntityBuffer.set(selectedEntity);
      return;
    }
    // If there is a selected entity and a buffer, and they are different, we should swap the selected entity and the buffer
    if (selectedEntity !== null && selectedEntityBuffer !== null && selectedEntity.id !== selectedEntityBuffer.id) {
      $selectedEntityBuffer.set(selectedEntity);
      dispatch(entitySelected({ entityIdentifier: selectedEntityBuffer }));
      return;
    }
  }, [dispatch, selectedEntity, selectedEntityBuffer]);

  useEffect(() => {
    if (!doesBufferExist) {
      $selectedEntityBuffer.set(null);
    }
  }, [doesBufferExist]);

  useHotkeys('q', quickSwitch, { enabled: true, preventDefault: true }, [quickSwitch]);
};
