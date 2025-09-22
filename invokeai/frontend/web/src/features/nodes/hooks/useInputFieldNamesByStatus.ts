import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvocationNodeContext } from 'features/nodes/components/flow/nodes/Invocation/context';
import type { FieldInputTemplate } from 'features/nodes/types/field';
import { isSingleOrCollection, isStatefulFieldType } from 'features/nodes/types/field';
import { useMemo } from 'react';

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
        const fieldNames: string[] = [];
        for (const [fieldName, fieldTemplate] of Object.entries(template?.inputs ?? {})) {
          if (isAnyOrDirectInputField(fieldTemplate)) {
            fieldNames.push(fieldName);
          }
        }
        return fieldNames;
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
        const fieldNames: string[] = [];
        for (const [fieldName, fieldTemplate] of Object.entries(template?.inputs ?? {})) {
          if (isConnectionInputField(fieldTemplate)) {
            fieldNames.push(fieldName);
          }
        }
        return fieldNames;
      }),
    [ctx]
  );
  return useAppSelector(selector);
};
