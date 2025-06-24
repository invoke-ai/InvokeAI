import { useAppSelector } from 'app/store/storeHooks';
import {
  selectFieldIdentifiersWithInvocationTypes,
  usePublishInputs,
} from 'features/nodes/components/sidePanel/workflow/publish';
import { useMemo } from 'react';

import { useInputFieldTemplateTitleOrThrow } from './useInputFieldTemplateTitleOrThrow';
import { useInputFieldUserTitleOrThrow } from './useInputFieldUserTitleOrThrow';

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
 * across the entire workflow. It uses a single-pass algorithm that groups by sanitized
 * base names and assigns unique names efficiently.
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

    // Create a lookup map for field identifiers to avoid repeated array searches
    const fieldIdentifierMap = new Map<string, { nodeId: string; fieldName: string; type: string; label?: string }>();
    for (const fieldIdentifier of fieldIdentifiersWithTypes) {
      const key = `${fieldIdentifier.nodeId}:${fieldIdentifier.fieldName}`;
      fieldIdentifierMap.set(key, fieldIdentifier);
    }

    // Group inputs by their sanitized base name in a single pass
    const baseNameGroups = new Map<string, Array<{ nodeId: string; fieldName: string; title: string }>>();

    for (const input of publishInputs.publishable) {
      const key = `${input.nodeId}:${input.fieldName}`;
      const fieldIdentifier = fieldIdentifierMap.get(key);

      // Get the title (user label or fallback to field name)
      const title = fieldIdentifier?.label || input.fieldName;
      const baseName = sanitizeFieldName(title);

      if (!baseNameGroups.has(baseName)) {
        baseNameGroups.set(baseName, []);
      }
      baseNameGroups.get(baseName)!.push({
        nodeId: input.nodeId,
        fieldName: input.fieldName,
        title,
      });
    }

    // Process each group and assign unique names
    for (const [baseName, inputs] of baseNameGroups) {
      if (inputs.length === 1) {
        // No conflict, use the base name
        const { nodeId, fieldName } = inputs[0];
        if (!fieldKeysMap.has(nodeId)) {
          fieldKeysMap.set(nodeId, new Map());
        }
        fieldKeysMap.get(nodeId)!.set(fieldName, baseName);
      } else {
        // Conflict detected, assign numbered names
        for (let i = 0; i < inputs.length; i++) {
          const { nodeId, fieldName } = inputs[i];
          const uniqueName = i === 0 ? baseName : `${baseName}_${i}`;

          if (!fieldKeysMap.has(nodeId)) {
            fieldKeysMap.set(nodeId, new Map());
          }
          fieldKeysMap.get(nodeId)!.set(fieldName, uniqueName);
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

/**
 * Hook that returns the sanitized field key for a specific node and field
 * @param nodeId The ID of the node
 * @param fieldName The name of the field
 * @returns The sanitized and deduplicated field key
 */
export const useInputFieldKey = (nodeId: string, fieldName: string) => {
  const allFieldKeys = useAllInputFieldKeys();
  const fieldUserTitle = useInputFieldUserTitleOrThrow(nodeId, fieldName);
  const fieldTemplateTitle = useInputFieldTemplateTitleOrThrow(nodeId, fieldName);

  return useMemo(() => {
    const nodeFieldKeys = allFieldKeys.get(nodeId);
    if (!nodeFieldKeys) {
      // Fallback to the old method if the field is not in publishable inputs
      return sanitizeFieldName(fieldUserTitle || fieldTemplateTitle);
    }

    const fieldKey = nodeFieldKeys.get(fieldName);
    if (!fieldKey) {
      // Fallback to the old method if the field is not in publishable inputs
      return sanitizeFieldName(fieldUserTitle || fieldTemplateTitle);
    }

    return fieldKey;
  }, [allFieldKeys, nodeId, fieldName, fieldUserTitle, fieldTemplateTitle]);
};
