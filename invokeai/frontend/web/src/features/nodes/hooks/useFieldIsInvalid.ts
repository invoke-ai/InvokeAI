import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useConnectionState } from 'features/nodes/hooks/useConnectionState';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import { selectFieldInputInstance, selectNodesSlice } from 'features/nodes/store/selectors';
import { isImageFieldCollectionInputInstance, isImageFieldCollectionInputTemplate } from 'features/nodes/types/field';
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
        // Image collections may have min or max item counts
        if (template.minItems !== undefined && field.value.length < template.minItems) {
          return true;
        }

        if (template.maxItems !== undefined && field.value.length > template.maxItems) {
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
