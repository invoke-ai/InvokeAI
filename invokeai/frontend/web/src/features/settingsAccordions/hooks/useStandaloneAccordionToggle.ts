import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { accordionStateChanged, selectUiSlice } from 'features/ui/store/uiSlice';
import { useCallback, useMemo } from 'react';

type UseStandaloneAccordionToggleArg = {
  defaultIsOpen: boolean;
  id: string;
};

export const useStandaloneAccordionToggle = (arg: UseStandaloneAccordionToggleArg) => {
  const dispatch = useAppDispatch();
  const selectIsOpen = useMemo(
    () => createSelector(selectUiSlice, (ui) => ui.accordions[arg.id] ?? arg.defaultIsOpen),
    [arg]
  );
  const isOpen = useAppSelector(selectIsOpen);
  const onToggle = useCallback(() => {
    dispatch(accordionStateChanged({ id: arg.id, isOpen: !isOpen }));
  }, [arg.id, dispatch, isOpen]);
  return { isOpen, onToggle };
};
