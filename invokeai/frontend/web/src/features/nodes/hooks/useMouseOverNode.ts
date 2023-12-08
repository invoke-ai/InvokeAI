import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { mouseOverNodeChanged } from 'features/nodes/store/nodesSlice';
import { useCallback, useMemo } from 'react';

export const useMouseOverNode = (nodeId: string) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        stateSelector,
        ({ nodes }) => nodes.mouseOverNode === nodeId
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
