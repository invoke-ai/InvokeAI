import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { FIELDS } from 'features/nodes/types/constants';
import { FieldType } from 'features/nodes/types/types';

export const getFieldColor = (fieldType: FieldType | string | null): string => {
  if (!fieldType) {
    return colorTokenToCssVar('base.500');
  }
  const color = FIELDS[fieldType]?.color;

  return color ? colorTokenToCssVar(color) : colorTokenToCssVar('base.500');
};
