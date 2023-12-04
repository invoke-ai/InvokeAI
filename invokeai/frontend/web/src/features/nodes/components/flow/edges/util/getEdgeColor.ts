import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { FIELD_COLORS } from 'features/nodes/types/constants';
import { FieldType } from 'features/nodes/types/field';

export const getFieldColor = (fieldType: FieldType | null): string => {
  if (!fieldType) {
    return colorTokenToCssVar('base.500');
  }
  const color = FIELD_COLORS[fieldType.name];

  return color ? colorTokenToCssVar(color) : colorTokenToCssVar('base.500');
};
