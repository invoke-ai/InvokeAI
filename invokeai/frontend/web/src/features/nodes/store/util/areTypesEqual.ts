import type { FieldType } from 'features/nodes/types/field';
import { isEqual, omit } from 'lodash-es';

/**
 * Checks if two types are equal. If the field types have original types, those are also compared. Any match is
 * considered equal. For example, if the source type and original target type match, the types are considered equal.
 * @param sourceType The type of the source field.
 * @param targetType The type of the target field.
 * @returns True if the types are equal, false otherwise.
 */

export const areTypesEqual = (sourceType: FieldType, targetType: FieldType) => {
  const _sourceType = 'originalType' in sourceType ? omit(sourceType, 'originalType') : sourceType;
  const _targetType = 'originalType' in targetType ? omit(targetType, 'originalType') : targetType;
  const _sourceTypeOriginal = 'originalType' in sourceType ? sourceType.originalType : null;
  const _targetTypeOriginal = 'originalType' in targetType ? targetType.originalType : null;
  if (isEqual(_sourceType, _targetType)) {
    return true;
  }
  if (_targetTypeOriginal && isEqual(_sourceType, _targetTypeOriginal)) {
    return true;
  }
  if (_sourceTypeOriginal && isEqual(_sourceTypeOriginal, _targetType)) {
    return true;
  }
  if (_sourceTypeOriginal && _targetTypeOriginal && isEqual(_sourceTypeOriginal, _targetTypeOriginal)) {
    return true;
  }
  return false;
};
