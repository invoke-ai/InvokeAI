import { useAppStore } from 'app/store/nanostores/store';
import { getFormFieldInitialValues as _getFormFieldInitialValues } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { useCallback } from 'react';

export const useGetFormFieldInitialValues = () => {
  const store = useAppStore();

  const getFormFieldInitialValues = useCallback(() => {
    const { nodes, form } = selectNodesSlice(store.getState());
    return _getFormFieldInitialValues(form, nodes);
  }, [store]);

  return getFormFieldInitialValues;
};
