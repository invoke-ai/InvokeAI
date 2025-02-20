import { useAppStore } from 'app/store/nanostores/store';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { selectFormInitialValues, selectNodeFieldElements } from 'features/nodes/store/workflowSlice';
import { useCallback } from 'react';

export const useResetAllNodeFields = () => {
  const store = useAppStore();
  const resetAllNodeValuesToDefaults = useCallback(() => {
    const allInitialValues = selectFormInitialValues(store.getState());
    const nodeFieldElements = selectNodeFieldElements(store.getState());
    for (const element of nodeFieldElements) {
      if (!(element.id in allInitialValues)) {
        continue;
      }
      const { nodeId, fieldName } = element.data.fieldIdentifier;
      store.dispatch(fieldValueReset({ nodeId, fieldName, value: allInitialValues[element.id] }));
    }
  }, [store]);
  return resetAllNodeValuesToDefaults;
};
