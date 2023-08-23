import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useCallback, useMemo } from 'react';
import { mouseOverFieldChanged } from '../store/nodesSlice';

export const useIsMouseOverField = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) =>
          nodes.mouseOverField?.nodeId === nodeId &&
          nodes.mouseOverField?.fieldName === fieldName,
        defaultSelectorOptions
      ),
    [fieldName, nodeId]
  );

  const isMouseOverField = useAppSelector(selector);

  const handleMouseOver = useCallback(() => {
    dispatch(mouseOverFieldChanged({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const handleMouseOut = useCallback(() => {
    dispatch(mouseOverFieldChanged(null));
  }, [dispatch]);

  return { isMouseOverField, handleMouseOver, handleMouseOut };
};
