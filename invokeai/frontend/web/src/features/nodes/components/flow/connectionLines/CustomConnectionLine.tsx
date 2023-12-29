import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import type { CSSProperties } from 'react';
import { memo } from 'react';
import type { ConnectionLineComponentProps } from 'reactflow';
import { getBezierPath } from 'reactflow';

const selector = createMemoizedSelector(stateSelector, ({ nodes }) => {
  const { shouldAnimateEdges, connectionStartFieldType, shouldColorEdges } =
    nodes;

  const stroke = shouldColorEdges
    ? getFieldColor(connectionStartFieldType)
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

const pathStyles: CSSProperties = { opacity: 0.8 };

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
        style={pathStyles}
      />
    </g>
  );
};

export default memo(CustomConnectionLine);
