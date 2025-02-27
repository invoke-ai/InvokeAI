import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { isSingleOrCollection } from 'features/nodes/types/field';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { useMemo } from 'react';

const isConnectionInputField = (field: FieldInputTemplate) => {
  return (
    (field.input === 'connection' && !isSingleOrCollection(field.type)) || !(field.type.name in TEMPLATE_BUILDER_MAP)
  );
};

const isAnyOrDirectInputField = (field: FieldInputTemplate) => {
  return (
    (['any', 'direct'].includes(field.input) || isSingleOrCollection(field.type)) &&
    field.type.name in TEMPLATE_BUILDER_MAP
  );
};

export const useInputFieldNamesMissing = (nodeId: string) => {
  const template = useNodeTemplate(nodeId);
  const node = useNodeData(nodeId);
  const fieldNames = useMemo(() => {
    const instanceFields = new Set(Object.keys(node.inputs));
    const allTemplateFields = new Set(Object.keys(template.inputs));
    return Array.from(instanceFields.difference(allTemplateFields));
  }, [node.inputs, template.inputs]);
  return fieldNames;
};

export const useInputFieldNamesAnyOrDirect = (nodeId: string) => {
  const template = useNodeTemplate(nodeId);
  const fieldNames = useMemo(() => {
    const anyOrDirectFields: string[] = [];
    for (const [fieldName, fieldTemplate] of Object.entries(template.inputs)) {
      if (isAnyOrDirectInputField(fieldTemplate)) {
        anyOrDirectFields.push(fieldName);
      }
    }
    return anyOrDirectFields;
  }, [template.inputs]);
  return fieldNames;
};

export const useInputFieldNamesConnection = (nodeId: string) => {
  const template = useNodeTemplate(nodeId);
  const fieldNames = useMemo(() => {
    const connectionFields: string[] = [];
    for (const [fieldName, fieldTemplate] of Object.entries(template.inputs)) {
      if (isConnectionInputField(fieldTemplate)) {
        connectionFields.push(fieldName);
      }
    }
    return connectionFields;
  }, [template.inputs]);
  return fieldNames;
};
