import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectActiveCanvasId, selectActiveTab } from 'features/controlLayers/store/selectors';
import { accordionStateChanged, selectUiSlice } from 'features/ui/store/uiSlice';
import { useCallback, useMemo } from 'react';

type UseStandaloneAccordionToggleArg = {
  defaultIsOpen: boolean;
  id: string;
};

export const useStandaloneAccordionToggle = (arg: UseStandaloneAccordionToggleArg) => {
  const dispatch = useAppDispatch();
  const activeTab = useAppSelector(selectActiveTab);
  const activeCanvasId = useAppSelector(selectActiveCanvasId);
  const accordionId = activeTab === 'canvas' ? `${activeCanvasId}-${arg.id}` : `${activeTab}-${arg.id}`;
  const selectIsOpen = useMemo(
    () => createSelector(selectUiSlice, (ui) => ui.accordions[accordionId] ?? arg.defaultIsOpen),
    [accordionId, arg]
  );
  const isOpen = useAppSelector(selectIsOpen);

  const onToggle = useCallback(() => {
    dispatch(accordionStateChanged({ id: accordionId, isOpen: !isOpen }));
  }, [dispatch, accordionId, isOpen]);

  return { isOpen, onToggle };
};
