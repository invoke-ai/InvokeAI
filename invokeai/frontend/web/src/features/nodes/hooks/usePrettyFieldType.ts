import type { FieldType } from 'features/nodes/types/field';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

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
    if (fieldType.isCollectionOrScalar) {
      return t('nodes.collectionOrScalarFieldType', { name });
    }
    return name;
  }, [fieldType, t]);

  return name;
};
