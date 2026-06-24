import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isNil } from 'es-toolkit/compat';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { isSingleOrCollection, isStatefulFieldType } from 'features/nodes/types/field';
import { useMemo } from 'react';

/**
 * Sort input fields: unordered fields first (preserving original order),
 * then explicitly ordered fields sorted by ui_order ascending.
 */
const sortInputFields = (fields: FieldInputTemplate[]): string[] => {
  const visibleFields = fields.filter((field) => !field.ui_hidden);

  const unorderedFields = visibleFields.filter((f) => isNil(f.ui_order));
  const orderedFields = visibleFields
    .filter((f) => !isNil(f.ui_order))
    .sort((a, b) => (a.ui_order ?? 0) - (b.ui_order ?? 0));

  return unorderedFields
    .concat(orderedFields)
    .map((f) => f.name)
    .filter((fieldName) => fieldName !== 'is_intermediate');
};

const isConnectionInputField = (field: FieldInputTemplate) => {
  return (field.input === 'connection' && !isSingleOrCollection(field.type)) || !isStatefulFieldType(field.type);
};

const isAnyOrDirectInputField = (field: FieldInputTemplate) => {
  return (
    (['any', 'direct'].includes(field.input) || isSingleOrCollection(field.type)) && isStatefulFieldType(field.type)
  );
};

export const useInputFieldNamesMissing = () => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector([ctx.selectNodeInputsOrThrow, ctx.selectNodeTemplateSafe], (inputs, template) => {
        const instanceFieldNames = new Set(Object.keys(inputs));
        const templateFieldNames = new Set(Object.keys(template?.inputs ?? {}));
        return Array.from(instanceFieldNames.difference(templateFieldNames));
      }),
    [ctx]
  );
  return useAppSelector(selector);
};

export const useInputFieldNamesAnyOrDirect = () => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector([ctx.selectNodeTemplateSafe], (template) => {
        const fields: FieldInputTemplate[] = [];
        for (const fieldTemplate of Object.values(template?.inputs ?? {})) {
          if (isAnyOrDirectInputField(fieldTemplate)) {
            fields.push(fieldTemplate);
          }
        }
        return sortInputFields(fields);
      }),
    [ctx]
  );
  return useAppSelector(selector);
};

export const useInputFieldNamesConnection = () => {
  const ctx = useInvocationNodeContext();
  const selector = useMemo(
    () =>
      createSelector([ctx.selectNodeTemplateSafe], (template) => {
        const fields: FieldInputTemplate[] = [];
        for (const fieldTemplate of Object.values(template?.inputs ?? {})) {
          if (isConnectionInputField(fieldTemplate)) {
            fields.push(fieldTemplate);
          }
        }
        return sortInputFields(fields);
      }),
    [ctx]
  );
  return useAppSelector(selector);
};
