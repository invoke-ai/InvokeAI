import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { formElementAdded, formElementRemoved } from 'features/nodes/store/nodesSlice';
import { buildSelectWorkflowFormNodeElement, selectFormRootElementId } from 'features/nodes/store/selectors';
import { buildNodeFieldElement } from 'features/nodes/types/workflow';
import { useCallback, useMemo } from 'react';

export const useAddRemoveFormElement = (nodeId: string, fieldName: string) => {
  const dispatch = useAppDispatch();
  const rootElementId = useAppSelector(selectFormRootElementId);
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldName);
  const field = useInputFieldInstance(fieldName);
  const selectWorkflowFormNodeElement = useMemo(
    () => buildSelectWorkflowFormNodeElement(nodeId, fieldName),
    [nodeId, fieldName]
  );
  const workflowFormNodeElement = useAppSelector(selectWorkflowFormNodeElement);
  const isAddedToRoot = useMemo(() => {
    return !!workflowFormNodeElement;
  }, [workflowFormNodeElement]);

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

  const removeNodeFieldFromRoot = useCallback(() => {
    if (!workflowFormNodeElement) {
      return;
    }
    dispatch(
      formElementRemoved({
        id: workflowFormNodeElement.id,
      })
    );
  }, [workflowFormNodeElement, dispatch]);

  return { isAddedToRoot, addNodeFieldToRoot, removeNodeFieldFromRoot };
};
