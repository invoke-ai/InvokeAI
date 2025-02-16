import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useInputFieldIsConnected } from 'features/nodes/hooks/useInputFieldIsConnected';
import { useInputFieldTemplate } from 'features/nodes/hooks/useInputFieldTemplate';
import { selectFieldInputInstance, selectNodesSlice } from 'features/nodes/store/selectors';
import {
  isFloatFieldCollectionInputInstance,
  isFloatFieldCollectionInputTemplate,
  isImageFieldCollectionInputInstance,
  isImageFieldCollectionInputTemplate,
  isIntegerFieldCollectionInputInstance,
  isIntegerFieldCollectionInputTemplate,
  isStringFieldCollectionInputInstance,
  isStringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import {
  validateImageFieldCollectionValue,
  validateNumberFieldCollectionValue,
  validateStringFieldCollectionValue,
} from 'features/nodes/types/fieldValidators';
import { useMemo } from 'react';

export const useInputFieldIsInvalid = (nodeId: string, fieldName: string) => {
  const template = useInputFieldTemplate(nodeId, fieldName);
  const isConnected = useInputFieldIsConnected(nodeId, fieldName);

  const selectIsInvalid = useMemo(() => {
    return createSelector(selectNodesSlice, (nodes) => {
      const field = selectFieldInputInstance(nodes, nodeId, fieldName);

      // No field instance is a problem - should not happen
      if (!field) {
        return true;
      }

      // 'connection' input fields have no data validation - only connection validation
      if (template.input === 'connection') {
        return template.required && !isConnected;
      }

      // 'any' input fields are valid if they are connected
      if (template.input === 'any' && isConnected) {
        return false;
      }

      // If there is no valid for the field & the field is required, it is invalid
      if (field.value === undefined) {
        return template.required;
      }

      // Else special handling for individual field types

      // Check the template type first - it's the most efficient. If that passes, check the instance type, which uses
      // zod and therefore is slower.

      if (isImageFieldCollectionInputTemplate(template) && isImageFieldCollectionInputInstance(field)) {
        if (validateImageFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      if (isStringFieldCollectionInputTemplate(template) && isStringFieldCollectionInputInstance(field)) {
        if (validateStringFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      if (isIntegerFieldCollectionInputTemplate(template) && isIntegerFieldCollectionInputInstance(field)) {
        if (validateNumberFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      if (isFloatFieldCollectionInputTemplate(template) && isFloatFieldCollectionInputInstance(field)) {
        if (validateNumberFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      // Field looks OK
      return false;
    });
  }, [nodeId, fieldName, template, isConnected]);

  const isInvalid = useAppSelector(selectIsInvalid);

  return isInvalid;
};
