import { isNil } from 'es-toolkit/compat';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import { isNodeAttributeFieldName } from 'features/nodes/types/nodeAttributeFields';

export const getSortedFilteredFieldNames = (fields: FieldInputTemplate[] | FieldOutputTemplate[]): string[] => {
  const visibleFields = fields.filter((field) => !field.ui_hidden);

  // we want explicitly ordered fields to be before unordered fields; split the list
  const orderedFields = visibleFields
    .filter((f) => !isNil(f.ui_order))
    .sort((a, b) => (a.ui_order ?? 0) - (b.ui_order ?? 0));
  const unorderedFields = visibleFields.filter((f) => isNil(f.ui_order));

  // concat the lists, and return the field names, skipping node attribute fields - they live in the node footer
  return orderedFields
    .concat(unorderedFields)
    .map((f) => f.name)
    .filter((fieldName) => !isNodeAttributeFieldName(fieldName));
};
