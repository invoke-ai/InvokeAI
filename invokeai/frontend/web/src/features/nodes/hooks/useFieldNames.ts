import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { useNodeTemplate } from 'features/nodes/hooks/useNodeTemplate';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { isSingleOrCollection } from 'features/nodes/types/field';
import { TEMPLATE_BUILDER_MAP } from 'features/nodes/util/schema/buildFieldInputTemplate';
import { difference, filter, keys } from 'lodash-es';
import { useMemo } from 'react';

const isConnectionInputField = (field: FieldInputTemplate) => {
  return (
    (field.input === 'connection' && !isSingleOrCollection(field.type)) ||
    !keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
  );
};

const isAnyOrDirectInputField = (field: FieldInputTemplate) => {
  return (
    (['any', 'direct'].includes(field.input) || isSingleOrCollection(field.type)) &&
    keys(TEMPLATE_BUILDER_MAP).includes(field.type.name)
  );
};

export const useFieldNames = (nodeId: string) => {
  const template = useNodeTemplate(nodeId);
  const node = useNodeData(nodeId);
  const fieldNames = useMemo(() => {
    const instanceFields = keys(node.inputs);
    const allTemplateFields = keys(template.inputs);
    const missingFields = difference(instanceFields, allTemplateFields);
    const connectionFields = filter(template.inputs, isConnectionInputField).map((f) => f.name);
    const anyOrDirectFields = filter(template.inputs, isAnyOrDirectInputField).map((f) => f.name);
    return {
      missingFields,
      connectionFields,
      anyOrDirectFields,
    };
  }, [node.inputs, template.inputs]);
  return fieldNames;
};
