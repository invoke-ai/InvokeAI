import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectActiveCanvasId, selectActiveTab } from 'features/controlLayers/store/selectors';
import { expanderStateChanged, selectUiSlice } from 'features/ui/store/uiSlice';
import { useCallback, useMemo } from 'react';

type UseExpanderToggleArg = {
  defaultIsOpen: boolean;
  id: string;
};

export const useExpanderToggle = (arg: UseExpanderToggleArg) => {
  const dispatch = useAppDispatch();
  const activeTab = useAppSelector(selectActiveTab);
  const activeCanvasId = useAppSelector(selectActiveCanvasId);
  const expanderId = activeTab === 'canvas' ? `${activeCanvasId}-${arg.id}` : `${activeTab}-${arg.id}`;
  const selectIsOpen = useMemo(
    () => createSelector(selectUiSlice, (ui) => ui.expanders[expanderId] ?? arg.defaultIsOpen),
    [expanderId, arg]
  );
  const isOpen = useAppSelector(selectIsOpen);

  const onToggle = useCallback(() => {
    dispatch(expanderStateChanged({ id: expanderId, isOpen: !isOpen }));
  }, [dispatch, expanderId, isOpen]);
  return { isOpen, onToggle };
};
