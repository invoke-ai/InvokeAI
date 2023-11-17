import {
  COLLECTION_MAP,
  COLLECTION_TYPES,
  POLYMORPHIC_TO_SINGLE_MAP,
  POLYMORPHIC_TYPES,
} from 'features/nodes/types/constants';
import { FieldType } from 'features/nodes/types/types';

export const validateSourceAndTargetTypes = (
  sourceType: FieldType | string,
  targetType: FieldType | string
) => {
  // TODO: There's a bug with Collect -> Iterate nodes:
  // https://github.com/invoke-ai/InvokeAI/issues/3956
  // Once this is resolved, we can remove this check.
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
    sourceType === 'CollectionItem' &&
    !COLLECTION_TYPES.some((t) => t === targetType);

  const isNonCollectionToCollectionItem =
    targetType === 'CollectionItem' &&
    !COLLECTION_TYPES.some((t) => t === sourceType) &&
    !POLYMORPHIC_TYPES.some((t) => t === sourceType);

  const isAnythingToPolymorphicOfSameBaseType =
    POLYMORPHIC_TYPES.some((t) => t === targetType) &&
    (() => {
      if (!POLYMORPHIC_TYPES.some((t) => t === targetType)) {
        return false;
      }
      const baseType =
        POLYMORPHIC_TO_SINGLE_MAP[
          targetType as keyof typeof POLYMORPHIC_TO_SINGLE_MAP
        ];

      const collectionType =
        COLLECTION_MAP[baseType as keyof typeof COLLECTION_MAP];

      return sourceType === baseType || sourceType === collectionType;
    })();

  const isGenericCollectionToAnyCollectionOrPolymorphic =
    sourceType === 'Collection' &&
    (COLLECTION_TYPES.some((t) => t === targetType) ||
      POLYMORPHIC_TYPES.some((t) => t === targetType));

  const isCollectionToGenericCollection =
    targetType === 'Collection' &&
    COLLECTION_TYPES.some((t) => t === sourceType);

  const isIntToFloat = sourceType === 'integer' && targetType === 'float';

  const isIntOrFloatToString =
    (sourceType === 'integer' || sourceType === 'float') &&
    targetType === 'string';

  const isTargetAnyType = targetType === 'Any';

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
