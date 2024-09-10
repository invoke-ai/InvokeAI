import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import { selectAllEntities, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasEntityState } from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

const selectNextEntityIdentifier = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  const selectedEntityIdentifier = canvas.selectedEntityIdentifier;
  const allEntities = selectAllEntities(canvas);
  let nextEntity: CanvasEntityState | null = null;
  if (!selectedEntityIdentifier) {
    nextEntity = allEntities[0] ?? null;
  } else {
    const selectedEntityIndex = allEntities.findIndex((entity) => entity.id === selectedEntityIdentifier.id);
    nextEntity = allEntities[(selectedEntityIndex + 1) % allEntities.length] ?? null;
  }
  if (!nextEntity) {
    return null;
  }
  return getEntityIdentifier(nextEntity);
});

const selectPrevEntityIdentifier = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  const selectedEntityIdentifier = canvas.selectedEntityIdentifier;
  const allEntities = selectAllEntities(canvas);
  let prevEntity: CanvasEntityState | null = null;
  if (!selectedEntityIdentifier) {
    prevEntity = allEntities[0] ?? null;
  } else {
    const selectedEntityIndex = allEntities.findIndex((entity) => entity.id === selectedEntityIdentifier.id);
    prevEntity = allEntities[(selectedEntityIndex - 1 + allEntities.length) % allEntities.length] ?? null;
  }
  if (!prevEntity) {
    return null;
  }
  return getEntityIdentifier(prevEntity);
});

export const useNextPrevEntityHotkeys = () => {
  useAssertSingleton('useNextPrevEntityHotkeys');
  const dispatch = useAppDispatch();

  const nextEntityIdentifier = useAppSelector(selectNextEntityIdentifier);
  const prevEntityIdentifier = useAppSelector(selectPrevEntityIdentifier);

  const selectNextEntity = useCallback(() => {
    if (nextEntityIdentifier) {
      dispatch(entitySelected({ entityIdentifier: nextEntityIdentifier }));
    }
  }, [dispatch, nextEntityIdentifier]);

  const selectPrevEntity = useCallback(() => {
    if (prevEntityIdentifier) {
      dispatch(entitySelected({ entityIdentifier: prevEntityIdentifier }));
    }
  }, [dispatch, prevEntityIdentifier]);

  useHotkeys(
    // “ === alt+[
    ['“'],
    selectPrevEntity,
    { preventDefault: true, ignoreModifiers: true },
    [selectPrevEntity]
  );

  useHotkeys(['alt+['], selectPrevEntity, { preventDefault: true }, [selectPrevEntity]);
  useHotkeys(
    // ‘ === alt+]
    ['‘'],
    selectNextEntity,
    { preventDefault: true, ignoreModifiers: true },
    [selectNextEntity]
  );
  useHotkeys(['alt+]'], selectNextEntity, { preventDefault: true }, [selectNextEntity]);
};
