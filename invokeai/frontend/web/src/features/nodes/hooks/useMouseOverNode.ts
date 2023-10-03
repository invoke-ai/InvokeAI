import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useCallback, useMemo } from 'react';
import { mouseOverNodeChanged } from '../store/nodesSlice';

export const useMouseOverNode = (nodeId: string) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createSelector(
        stateSelector,
        ({ nodes }) => nodes.mouseOverNode === nodeId,
        defaultSelectorOptions
      ),
    [nodeId]
  );

  const isMouseOverNode = useAppSelector(selector);

  const handleMouseOver = useCallback(() => {
    !isMouseOverNode && dispatch(mouseOverNodeChanged(nodeId));
  }, [dispatch, nodeId, isMouseOverNode]);

  const handleMouseOut = useCallback(() => {
    isMouseOverNode && dispatch(mouseOverNodeChanged(null));
  }, [dispatch, isMouseOverNode]);

  return { isMouseOverNode, handleMouseOver, handleMouseOut };
};
