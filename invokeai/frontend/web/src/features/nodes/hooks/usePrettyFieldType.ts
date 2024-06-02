import { type FieldType, isCollection, isSingleOrCollection } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export const useFieldTypeName = (fieldType?: FieldType): string => {
  const { t } = useTranslation();

  const name = useMemo(() => {
    if (!fieldType) {
      return '';
    }
    const { name } = fieldType;
    if (isCollection(fieldType)) {
      return t('nodes.collectionFieldType', { name });
    }
    if (isSingleOrCollection(fieldType)) {
      return t('nodes.collectionOrScalarFieldType', { name });
    }
    return t('nodes.singleFieldType', { name });
  }, [fieldType, t]);

  return name;
};
