import { useAppSelector } from 'app/store/storeHooks';
import {
  selectFieldIdentifiersWithInvocationTypes,
  usePublishInputs,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { useMemo } from 'react';

// Helper function to sanitize a field name
const sanitizeFieldName = (name: string): string => {
  return name
    .toLowerCase() // Convert to lowercase
    .replace(/\s+/g, '_') // Replace spaces with underscores
    .replace(/[^a-z0-9_]+/g, ''); // Remove any non-alphanumeric characters
};

/**
 * Hook that calculates all sanitized and deduplicated field keys for publishable inputs.
 *
 * This hook processes all publishable inputs at once to ensure field name uniqueness
 * across the entire workflow. It uses a grouping algorithm:
 * 1. Group all inputs by their sanitized base name
 * 2. For each group, assign unique names (base name for single items, numbered for conflicts)
 *
 * @returns A map of nodeId -> fieldName -> sanitized field key
 *
 * @example
 * ```tsx
 * const allFieldKeys = useAllInputFieldKeys();
 * const fieldKey = allFieldKeys.get(nodeId)?.get(fieldName);
 * ```
 */
export const useAllInputFieldKeys = () => {
  const publishInputs = usePublishInputs();
  const fieldIdentifiersWithTypes = useAppSelector(selectFieldIdentifiersWithInvocationTypes);

  return useMemo(() => {
    const fieldKeysMap = new Map<string, Map<string, string>>();

    // Group inputs by their sanitized base name
    const inputsByBaseName = new Map<
      string,
      Array<{
        input: (typeof publishInputs.publishable)[0];
        fieldIdentifier: (typeof fieldIdentifiersWithTypes)[0] | undefined;
        title: string;
        baseName: string;
      }>
    >();

    // First pass: group all inputs by their base sanitized name
    for (const input of publishInputs.publishable) {
      const fieldIdentifier = fieldIdentifiersWithTypes.find(
        (fi) => fi.nodeId === input.nodeId && fi.fieldName === input.fieldName
      );

      const title = fieldIdentifier?.label || input.fieldName;
      const baseName = sanitizeFieldName(title);

      if (!inputsByBaseName.has(baseName)) {
        inputsByBaseName.set(baseName, []);
      }
      inputsByBaseName.get(baseName)!.push({
        input,
        fieldIdentifier,
        title,
        baseName,
      });
    }

    // Second pass: process each group and assign unique names
    for (const [baseName, inputs] of inputsByBaseName) {
      if (inputs.length === 1) {
        // No conflict, use the base name
        const input = inputs[0];
        if (!input) {
          continue; // Skip if input is undefined
        }
        if (!fieldKeysMap.has(input.input.nodeId)) {
          fieldKeysMap.set(input.input.nodeId, new Map());
        }
        fieldKeysMap.get(input.input.nodeId)!.set(input.input.fieldName, baseName);
      } else {
        // Conflict detected, assign numbered names
        for (let i = 0; i < inputs.length; i++) {
          const input = inputs[i];
          if (!input) {
            continue; // Skip if input is undefined
          }
          const uniqueName = i === 0 ? baseName : `${baseName}_${i}`;

          if (!fieldKeysMap.has(input.input.nodeId)) {
            fieldKeysMap.set(input.input.nodeId, new Map());
          }
          fieldKeysMap.get(input.input.nodeId)!.set(input.input.fieldName, uniqueName);
        }
      }
    }

    return fieldKeysMap;
  }, [publishInputs, fieldIdentifiersWithTypes]);
};

/**
 * Helper function to get a field key from the map with fallback.
 *
 * This function is useful when you already have the `allFieldKeys` map
 * and want to get a specific field key without calling hooks again.
 *
 * @param allFieldKeys The map of all field keys
 * @param nodeId The ID of the node
 * @param fieldName The name of the field
 * @returns The sanitized field key
 *
 * @example
 * ```tsx
 * const allFieldKeys = useAllInputFieldKeys();
 * const fieldKey = getFieldKeyFromMap(allFieldKeys, nodeId, fieldName, fallbackTitle);
 * ```
 */
export const getFieldKeyFromMap = (
  allFieldKeys: Map<string, Map<string, string>>,
  nodeId: string,
  fieldName: string
): string => {
  const nodeFieldKeys = allFieldKeys.get(nodeId);
  if (!nodeFieldKeys) {
    throw new Error(`Node ${nodeId} not found in field keys map`);
  }

  const fieldKey = nodeFieldKeys.get(fieldName);
  if (!fieldKey) {
    throw new Error(`Field ${fieldName} not found in node ${nodeId} field keys map`);
  }

  return fieldKey;
};
