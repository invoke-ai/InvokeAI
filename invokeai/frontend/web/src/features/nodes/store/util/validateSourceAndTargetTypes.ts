import { FieldType } from 'features/nodes/types/types';
import {
  getBaseType,
  getIsCollection,
  getIsPolymorphic,
} from './parseFieldType';

/**
 * Validates that the source and target types are compatible for a connection.
 * @param sourceType The type of the source field. Must be the originalType if it exists.
 * @param targetType The type of the target field. Must be the originalType if it exists.
 * @returns True if the connection is valid, false otherwise.
 */
export const validateSourceAndTargetTypes = (
  sourceType: FieldType | string,
  targetType: FieldType | string
) => {
  const isSourcePolymorphic = getIsPolymorphic(sourceType);
  const isSourceCollection = getIsCollection(sourceType);
  const sourceBaseType = getBaseType(sourceType);

  const isTargetPolymorphic = getIsPolymorphic(targetType);
  const isTargetCollection = getIsCollection(targetType);
  const targetBaseType = getBaseType(targetType);

  // TODO: There's a bug with Collect -> Iterate nodes:
  // https://github.com/invoke-ai/InvokeAI/issues/3956
  // Once this is resolved, we can remove this check.
  // Note that 'Collection' here is a field type, not node type.
  if (sourceType === 'Collection' && targetType === 'Collection') {
    return false;
  }

  if (sourceType === targetType) {
    return true;
  }

  /**
   * Connection types must be the same for a connection, with exceptions:
   * - CollectionItem can connect to any non-Collection
   * - Non-Collections can connect to CollectionItem
   * - Anything (non-Collections, Collections, Polymorphics) can connect to Polymorphics of the same base type
   * - Generic Collection can connect to any other Collection or Polymorphic
   * - Any Collection can connect to a Generic Collection
   */

  const isCollectionItemToNonCollection =
    sourceType === 'CollectionItem' && !isTargetCollection;

  const isNonCollectionToCollectionItem =
    targetType === 'CollectionItem' &&
    !isSourceCollection &&
    !isSourcePolymorphic;

  const isAnythingToPolymorphicOfSameBaseType =
    isTargetPolymorphic && sourceBaseType === targetBaseType;

  const isGenericCollectionToAnyCollectionOrPolymorphic =
    sourceType === 'Collection' && (isTargetCollection || isTargetPolymorphic);

  const isCollectionToGenericCollection =
    targetType === 'Collection' && isSourceCollection;

  const isIntToFloat = sourceType === 'integer' && targetType === 'float';

  const isIntOrFloatToString =
    (sourceType === 'integer' || sourceType === 'float') &&
    targetType === 'string';

  const isTargetAnyType = targetType === 'Any';

  // One of these must be true for the connection to be valid
  return (
    isCollectionItemToNonCollection ||
    isNonCollectionToCollectionItem ||
    isAnythingToPolymorphicOfSameBaseType ||
    isGenericCollectionToAnyCollectionOrPolymorphic ||
    isCollectionToGenericCollection ||
    isIntToFloat ||
    isIntOrFloatToString ||
    isTargetAnyType
  );
};
