import type { FieldType } from 'features/nodes/types/field';
import { isEqual, omit } from 'lodash-es';

/**
 * Checks if two types are equal. If the field types have original types, those are also compared. Any match is
 * considered equal. For example, if the first type and original second type match, the types are considered equal.
 * @param firstType The first type to compare.
 * @param secondType The second type to compare.
 * @returns True if the types are equal, false otherwise.
 */
export const areTypesEqual = (firstType: FieldType, secondType: FieldType) => {
  const _firstType = 'originalType' in firstType ? omit(firstType, 'originalType') : firstType;
  const _secondType = 'originalType' in secondType ? omit(secondType, 'originalType') : secondType;
  const _originalFirstType = 'originalType' in firstType ? firstType.originalType : null;
  const _originalSecondType = 'originalType' in secondType ? secondType.originalType : null;
  if (isEqual(_firstType, _secondType)) {
    return true;
  }
  if (_originalSecondType && isEqual(_firstType, _originalSecondType)) {
    return true;
  }
  if (_originalFirstType && isEqual(_originalFirstType, _secondType)) {
    return true;
  }
  if (_originalFirstType && _originalSecondType && isEqual(_originalFirstType, _originalSecondType)) {
    return true;
  }
  return false;
};
