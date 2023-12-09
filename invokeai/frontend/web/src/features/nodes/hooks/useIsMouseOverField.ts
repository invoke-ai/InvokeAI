import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { mouseOverFieldChanged } from 'features/nodes/store/nodesSlice';
import { useCallback, useMemo } from 'react';

export const useIsMouseOverField = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const selector = useMemo(
    () =>
      createMemoizedSelector(
        stateSelector,
        ({ nodes }) =>
          nodes.mouseOverField?.nodeId === nodeId &&
          nodes.mouseOverField?.fieldName === fieldName
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
