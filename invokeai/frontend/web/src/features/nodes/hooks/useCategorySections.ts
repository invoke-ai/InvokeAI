import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { categorySectionsChanged, selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import { useCallback, useMemo } from 'react';

export const useCategorySections = (id: string) => {
  const dispatch = useAppDispatch();
  const selectIsOpen = useMemo(
    () => createSelector(selectWorkflowSlice, (workflow) => workflow.categorySections[id] ?? true),
    [id]
  );
  const isOpen = useAppSelector(selectIsOpen);
  const onToggle = useCallback(() => {
    dispatch(categorySectionsChanged({ id, isOpen: !isOpen }));
  }, [id, dispatch, isOpen]);

  return { isOpen, onToggle };
};
