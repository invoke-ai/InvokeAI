import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import { selectFieldInputInstance, selectNodesSlice } from 'features/nodes/store/selectors';
import {
  isImageFieldCollectionInputInstance,
  isImageFieldCollectionInputTemplate,
  isIntegerFieldCollectionInputInstance,
  isIntegerFieldCollectionInputTemplate,
  isStringFieldCollectionInputInstance,
  isStringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import { isNil } from 'lodash-es';
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

      // Image collections may have min or max item counts
      if (isImageFieldCollectionInputInstance(field) && isImageFieldCollectionInputTemplate(template)) {
        if (template.minItems !== undefined && field.value.length < template.minItems) {
          return true;
        }

        if (template.maxItems !== undefined && field.value.length > template.maxItems) {
          return true;
        }
      }

      // String collections may have min or max item counts
      if (isStringFieldCollectionInputInstance(field) && isStringFieldCollectionInputTemplate(template)) {
        if (template.minItems !== undefined && field.value.length < template.minItems) {
          return true;
        }

        if (template.maxItems !== undefined && field.value.length > template.maxItems) {
          return true;
        }
        if (field.value) {
          for (const str of field.value) {
            if (!isNil(template.maxLength) && str.length > template.maxLength) {
              return true;
            }
            if (!isNil(template.minLength) && str.length < template.minLength) {
              return true;
            }
          }
        }
      }

      // Integer collections may have min or max item counts
      if (isIntegerFieldCollectionInputInstance(field) && isIntegerFieldCollectionInputTemplate(template)) {
        if (template.minItems !== undefined && field.value.length < template.minItems) {
          return true;
        }

        if (template.maxItems !== undefined && field.value.length > template.maxItems) {
          return true;
        }
        if (field.value) {
          for (const int of field.value) {
            if (!isNil(template.maximum) && int > template.maximum) {
              return true;
            }
            if (!isNil(template.exclusiveMaximum) && int >= template.exclusiveMaximum) {
              return true;
            }
            if (!isNil(template.minimum) && int < template.minimum) {
              return true;
            }
            if (!isNil(template.exclusiveMinimum) && int <= template.exclusiveMinimum) {
              return true;
            }
          }
        }
      }

      // Field looks OK
      return false;
    });
  }, [connectionState.isConnected, fieldName, nodeId, template]);

  const isInvalid = useAppSelector(selectIsInvalid);

  return isInvalid;
};
