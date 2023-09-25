import {
  COLLECTION_MAP,
  COLLECTION_TYPES,
  POLYMORPHIC_TO_SINGLE_MAP,
  POLYMORPHIC_TYPES,
} from 'features/nodes/types/constants';
import { FieldType } from 'features/nodes/types/types';

export const validateSourceAndTargetTypes = (
  sourceType: FieldType,
  targetType: FieldType
) => {
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
    sourceType === 'CollectionItem' && !COLLECTION_TYPES.includes(targetType);

  const isNonCollectionToCollectionItem =
    targetType === 'CollectionItem' &&
    !COLLECTION_TYPES.includes(sourceType) &&
    !POLYMORPHIC_TYPES.includes(sourceType);

  const isAnythingToPolymorphicOfSameBaseType =
    POLYMORPHIC_TYPES.includes(targetType) &&
    (() => {
      if (!POLYMORPHIC_TYPES.includes(targetType)) {
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
    (COLLECTION_TYPES.includes(targetType) ||
      POLYMORPHIC_TYPES.includes(targetType));

  const isCollectionToGenericCollection =
    targetType === 'Collection' && COLLECTION_TYPES.includes(sourceType);

  const isIntToFloat = sourceType === 'integer' && targetType === 'float';

  const isIntOrFloatToString =
    (sourceType === 'integer' || sourceType === 'float') &&
    targetType === 'string';

  return (
    isCollectionItemToNonCollection ||
    isNonCollectionToCollectionItem ||
    isAnythingToPolymorphicOfSameBaseType ||
    isGenericCollectionToAnyCollectionOrPolymorphic ||
    isCollectionToGenericCollection ||
    isIntToFloat ||
    isIntOrFloatToString
  );
};
