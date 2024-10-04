import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { expanderStateChanged, selectUiSlice } from 'features/ui/store/uiSlice';
import { useCallback, useMemo } from 'react';

type UseExpanderToggleArg = {
  defaultIsOpen: boolean;
  id: string;
};

export const useExpanderToggle = (arg: UseExpanderToggleArg) => {
  const dispatch = useAppDispatch();
  const selectIsOpen = useMemo(
    () => createSelector(selectUiSlice, (ui) => ui.expanders[arg.id] ?? arg.defaultIsOpen),
    [arg]
  );
  const isOpen = useAppSelector(selectIsOpen);
  const onToggle = useCallback(() => {
    dispatch(expanderStateChanged({ id: arg.id, isOpen: !isOpen }));
  }, [dispatch, arg.id, isOpen]);
  return { isOpen, onToggle };
};
