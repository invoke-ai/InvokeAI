import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { formElementAdded, selectFormRootElementId } from 'features/nodes/store/workflowSlice';
import { buildNodeFieldElement } from 'features/nodes/types/workflow';
import { useCallback } from 'react';

export const useAddNodeFieldToRoot = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const rootElementId = useAppSelector(selectFormRootElementId);
  const fieldTemplate = useInputFieldTemplate(nodeId, fieldName);
  const field = useInputFieldInstance(nodeId, fieldName);

  const addNodeFieldToRoot = useCallback(() => {
    const element = buildNodeFieldElement(nodeId, fieldName, fieldTemplate.type);
    dispatch(
      formElementAdded({
        element,
        parentId: rootElementId,
        initialValue: field.value,
      })
    );
  }, [nodeId, fieldName, fieldTemplate.type, dispatch, rootElementId, field.value]);

  return addNodeFieldToRoot;
};
