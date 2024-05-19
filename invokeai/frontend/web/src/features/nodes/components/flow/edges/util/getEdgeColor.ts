import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { FIELD_COLORS } from 'features/nodes/types/constants';
import type { FieldType } from 'features/nodes/types/field';
import type { CSSProperties } from 'react';

export const getFieldColor = (fieldType: FieldType | null): string => {
  if (!fieldType) {
    return colorTokenToCssVar('base.500');
  }
  const color = FIELD_COLORS[fieldType.name];

  return color ? colorTokenToCssVar(color) : colorTokenToCssVar('base.500');
};

export const getEdgeStyles = (
  stroke: string,
  selected: boolean,
  shouldAnimateEdges: boolean,
  areConnectedNodesSelected: boolean
): CSSProperties => ({
  strokeWidth: 3,
  stroke,
  opacity: selected ? 1 : 0.5,
  animation: shouldAnimateEdges ? 'dashdraw 0.5s linear infinite' : undefined,
  strokeDasharray: selected || areConnectedNodesSelected ? 5 : 'none',
});
