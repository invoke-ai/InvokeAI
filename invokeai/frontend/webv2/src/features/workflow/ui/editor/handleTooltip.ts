import type { FieldType } from '@features/workflow/contracts';

import { getFieldTypeLabel } from '@features/workflow/utility';

export const getHandleTypeTooltip = (type: FieldType | null, fallback = 'Any'): string => {
  if (!type) {
    return fallback;
  }

  const label = getFieldTypeLabel(type);

  return type.batch ? `${label} batch` : label;
};
