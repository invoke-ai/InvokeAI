import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
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

export const useFieldIsInvalid = (nodeId: string, fieldName: string) => {
  const template = useFieldInputTemplate(nodeId, fieldName);
  const connectionState = useConnectionState({ nodeId, fieldName, kind: 'inputs' });

  const selectIsInvalid = useMemo(() => {
    return createSelector(selectNodesSlice, (nodes) => {
      const field = selectFieldInputInstance(nodes, nodeId, fieldName);

      // No field instance is a problem - should not happen
      if (!field) {
        return true;
      }

      // 'connection' input fields have no data validation - only connection validation
      if (template.input === 'connection') {
        return template.required && !connectionState.isConnected;
      }

      // 'any' input fields are valid if they are connected
      if (template.input === 'any' && connectionState.isConnected) {
        return false;
      }

      // If there is no valid for the field & the field is required, it is invalid
      if (field.value === undefined) {
        return template.required;
      }

      // Else special handling for individual field types

      if (isImageFieldCollectionInputInstance(field) && isImageFieldCollectionInputTemplate(template)) {
        if (validateImageFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      if (isStringFieldCollectionInputInstance(field) && isStringFieldCollectionInputTemplate(template)) {
        if (validateStringFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      if (isIntegerFieldCollectionInputInstance(field) && isIntegerFieldCollectionInputTemplate(template)) {
        if (validateNumberFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      if (isFloatFieldCollectionInputInstance(field) && isFloatFieldCollectionInputTemplate(template)) {
        if (validateNumberFieldCollectionValue(field.value, template).length > 0) {
          return true;
        }
      }

      // Field looks OK
      return false;
    });
  }, [connectionState.isConnected, fieldName, nodeId, template]);

  const isInvalid = useAppSelector(selectIsInvalid);

  return isInvalid;
};
