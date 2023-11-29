import { startCase } from 'lodash-es';
import { FieldOutputTemplate, FieldType } from 'features/nodes/types/field';
import { InvocationFieldSchema } from 'features/nodes/types/openapi';

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
