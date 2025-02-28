import { useAppStore } from 'app/store/nanostores/store';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import {
  getFormFieldInitialValues as _getFormFieldInitialValues,
  selectWorkflowForm,
} from 'features/nodes/store/workflowSlice';
import { useCallback } from 'react';

export const useGetFormFieldInitialValues = () => {
  const store = useAppStore();

  const getFormFieldInitialValues = useCallback(() => {
    const form = selectWorkflowForm(store.getState());
    const { nodes } = selectNodesSlice(store.getState());
    return _getFormFieldInitialValues(form, nodes);
  }, [store]);

  return getFormFieldInitialValues;
};
