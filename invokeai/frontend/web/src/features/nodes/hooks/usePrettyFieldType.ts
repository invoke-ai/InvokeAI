import { useTranslation } from 'react-i18next';
import { FieldType } from '../types/field';
import { useMemo } from 'react';

export const useFieldTypeName = (fieldType?: FieldType): string => {
  const { t } = useTranslation();

  const name = useMemo(() => {
    if (!fieldType) {
      return '';
    }
    const { name } = fieldType;
    if (fieldType.isCollection) {
      return t('nodes.collectionFieldType', { name });
    }
    if (fieldType.isPolymorphic) {
      return t('nodes.polymorphicFieldType', { name });
    }
    return name;
  }, [fieldType, t]);

  return name;
};
