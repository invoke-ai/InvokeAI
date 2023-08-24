import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { FIELDS } from 'features/nodes/types/constants';
import { memo } from 'react';
import { ConnectionLineComponentProps, getBezierPath } from 'reactflow';

const selector = createSelector(stateSelector, ({ nodes }) => {
  const { shouldAnimateEdges, currentConnectionFieldType, shouldColorEdges } =
    nodes;

  const stroke =
    currentConnectionFieldType && shouldColorEdges
      ? colorTokenToCssVar(FIELDS[currentConnectionFieldType].color)
      : colorTokenToCssVar('base.500');

  let className = 'react-flow__custom_connection-path';

  if (shouldAnimateEdges) {
    className = className.concat(' animated');
  }

  return {
    stroke,
    className,
  };
});

const CustomConnectionLine = ({
  fromX,
  fromY,
  fromPosition,
  toX,
  toY,
  toPosition,
}: ConnectionLineComponentProps) => {
  const { stroke, className } = useAppSelector(selector);

  const pathParams = {
    sourceX: fromX,
    sourceY: fromY,
    sourcePosition: fromPosition,
    targetX: toX,
    targetY: toY,
    targetPosition: toPosition,
  };

  const [dAttr] = getBezierPath(pathParams);

  return (
    <g>
      <path
        fill="none"
        stroke={stroke}
        strokeWidth={2}
        className={className}
        d={dAttr}
        style={{ opacity: 0.8 }}
      />
    </g>
  );
};

export default memo(CustomConnectionLine);
