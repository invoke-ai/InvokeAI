import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectInvocationNodeType, selectNodesSlice } from 'features/nodes/store/selectors';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { assert } from 'tsafe';

export const useFieldTemplate = (
  nodeId: string,
  fieldName: string,
  kind: 'inputs' | 'outputs'
): FieldInputTemplate | FieldOutputTemplate => {
  const templates = useStore($templates);
  const selectNodeType = useMemo(
    () => createSelector(selectNodesSlice, (nodes) => selectInvocationNodeType(nodes, nodeId)),
    [nodeId]
  );
  const nodeType = useAppSelector(selectNodeType);
  const fieldTemplate = useMemo(() => {
    const template = templates[nodeType];
    assert(template, `Template for node type ${nodeType} not found`);
    if (kind === 'inputs') {
      const fieldTemplate = template.inputs[fieldName];
      assert(fieldTemplate, `Field template for field ${fieldName} not found`);
      return fieldTemplate;
    } else {
      const fieldTemplate = template.outputs[fieldName];
      assert(fieldTemplate, `Field template for field ${fieldName} not found`);
      return fieldTemplate;
    }
  }, [fieldName, kind, nodeType, templates]);

  return fieldTemplate;
};
