import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { CSSProperties } from 'react';
import { memo } from 'react';
import type { ConnectionLineComponentProps } from 'reactflow';
import { getBezierPath } from 'reactflow';

const selectStroke = createSelector(selectNodesSlice, (nodes) =>
  nodes.shouldColorEdges ? getFieldColor(nodes.connectionStartFieldType) : colorTokenToCssVar('base.500')
);

const selectClassName = createSelector(selectNodesSlice, (nodes) =>
  nodes.shouldAnimateEdges ? 'react-flow__custom_connection-path animated' : 'react-flow__custom_connection-path'
);

const pathStyles: CSSProperties = { opacity: 0.8 };

const CustomConnectionLine = ({ fromX, fromY, fromPosition, toX, toY, toPosition }: ConnectionLineComponentProps) => {
  const stroke = useAppSelector(selectStroke);
  const className = useAppSelector(selectClassName);

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
      <path fill="none" stroke={stroke} strokeWidth={2} className={className} d={dAttr} style={pathStyles} />
    </g>
  );
};

export default memo(CustomConnectionLine);
