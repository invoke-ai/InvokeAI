import type { FieldType } from '@workbench/workflows/types';

import { getFieldTypeLabel } from '@workbench/workflows/fields';

export const getHandleTypeTooltip = (type: FieldType | null, fallback = 'Any'): string => {
  if (!type) {
    return fallback;
  }

  const label = getFieldTypeLabel(type);

  return type.batch ? `${label} batch` : label;
};
