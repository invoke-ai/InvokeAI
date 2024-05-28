import { areTypesEqual } from 'features/nodes/store/util/areTypesEqual';
import { type FieldType, isCollection, isSingle, isSingleOrCollection } from 'features/nodes/types/field';

/**
 * Validates that the source and target types are compatible for a connection.
 * @param sourceType The type of the source field.
 * @param targetType The type of the target field.
 * @returns True if the connection is valid, false otherwise.
 */
export const validateConnectionTypes = (sourceType: FieldType, targetType: FieldType) => {
  // TODO: There's a bug with Collect -> Iterate nodes:
  // https://github.com/invoke-ai/InvokeAI/issues/3956
  // Once this is resolved, we can remove this check.
  if (sourceType.name === 'CollectionField' && targetType.name === 'CollectionField') {
    return false;
  }

  if (areTypesEqual(sourceType, targetType)) {
    return true;
  }

  /**
   * Connection types must be the same for a connection, with exceptions:
   * - CollectionItem can connect to any non-COLLECTION (e.g. SINGLE or SINGLE_OR_COLLECTION)
   * - SINGLE can connect to CollectionItem
   * - Anything (SINGLE, COLLECTION, SINGLE_OR_COLLECTION) can connect to SINGLE_OR_COLLECTION of the same base type
   * - Generic CollectionField can connect to any other COLLECTION or SINGLE_OR_COLLECTION
   * - Any COLLECTION can connect to a Generic Collection
   */
  const isCollectionItemToNonCollection = sourceType.name === 'CollectionItemField' && !isCollection(targetType);

  const isNonCollectionToCollectionItem = isSingle(sourceType) && targetType.name === 'CollectionItemField';

  const isAnythingToSingleOrCollectionOfSameBaseType =
    isSingleOrCollection(targetType) && sourceType.name === targetType.name;

  const isGenericCollectionToAnyCollectionOrSingleOrCollection =
    sourceType.name === 'CollectionField' && !isSingle(targetType);

  const isCollectionToGenericCollection = targetType.name === 'CollectionField' && isCollection(sourceType);

  const isSourceSingle = isSingle(sourceType);
  const isTargetSingle = isSingle(targetType);
  const isSingleToSingle = isSourceSingle && isTargetSingle;
  const isSingleToSingleOrCollection = isSourceSingle && isSingleOrCollection(targetType);
  const isCollectionToCollection = isCollection(sourceType) && isCollection(targetType);
  const isCollectionToSingleOrCollection = isCollection(sourceType) && isSingleOrCollection(targetType);
  const isSingleOrCollectionToSingleOrCollection = isSingleOrCollection(sourceType) && isSingleOrCollection(targetType);
  const doesCardinalityMatch =
    isSingleToSingle ||
    isCollectionToCollection ||
    isCollectionToSingleOrCollection ||
    isSingleOrCollectionToSingleOrCollection ||
    isSingleToSingleOrCollection;

  const isIntToFloat = sourceType.name === 'IntegerField' && targetType.name === 'FloatField';
  const isIntToString = sourceType.name === 'IntegerField' && targetType.name === 'StringField';
  const isFloatToString = sourceType.name === 'FloatField' && targetType.name === 'StringField';

  const isSubTypeMatch = doesCardinalityMatch && (isIntToFloat || isIntToString || isFloatToString);

  const isTargetAnyType = targetType.name === 'AnyField';

  // One of these must be true for the connection to be valid
  return (
    isCollectionItemToNonCollection ||
    isNonCollectionToCollectionItem ||
    isAnythingToSingleOrCollectionOfSameBaseType ||
    isGenericCollectionToAnyCollectionOrSingleOrCollection ||
    isCollectionToGenericCollection ||
    isSubTypeMatch ||
    isTargetAnyType
  );
};
