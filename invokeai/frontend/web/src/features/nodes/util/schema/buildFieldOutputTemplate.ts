import type { FieldOutputTemplate, FieldType } from 'features/nodes/types/field';
import type { InvocationFieldSchema } from 'features/nodes/types/openapi';
import { startCase } from 'lodash-es';

export const buildFieldOutputTemplate = (
  fieldSchema: InvocationFieldSchema,
  fieldName: string,
  fieldType: FieldType
): FieldOutputTemplate => {
  const { title, description, ui_hidden, ui_type, ui_order } = fieldSchema;

  const fieldOutputTemplate: FieldOutputTemplate = {
    fieldKind: 'output',
    name: fieldName,
    title: title ?? (fieldName ? startCase(fieldName) : ''),
    description: description ?? '',
    type: fieldType,
    ui_hidden,
    ui_type,
    ui_order,
  };

  return fieldOutputTemplate;
};
