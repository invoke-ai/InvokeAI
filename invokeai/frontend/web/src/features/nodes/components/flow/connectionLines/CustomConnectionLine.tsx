import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { getFieldColor } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { $pendingConnection } from 'features/nodes/store/nodesSlice';
import { selectShouldAnimateEdges, selectShouldColorEdges } from 'features/nodes/store/workflowSettingsSlice';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import type { ConnectionLineComponentProps } from 'reactflow';
import { getBezierPath } from 'reactflow';

const pathStyles: CSSProperties = { opacity: 0.8 };

const CustomConnectionLine = ({ fromX, fromY, fromPosition, toX, toY, toPosition }: ConnectionLineComponentProps) => {
  const pendingConnection = useStore($pendingConnection);
  const shouldColorEdges = useAppSelector(selectShouldColorEdges);
  const shouldAnimateEdges = useAppSelector(selectShouldAnimateEdges);
  const stroke = useMemo(() => {
    if (shouldColorEdges && pendingConnection) {
      return getFieldColor(pendingConnection.fieldTemplate.type);
    } else {
      return colorTokenToCssVar('base.500');
    }
  }, [pendingConnection, shouldColorEdges]);
  const className = useMemo(() => {
    if (shouldAnimateEdges) {
      return 'react-flow__custom_connection-path animated';
    } else {
      return 'react-flow__custom_connection-path';
    }
  }, [shouldAnimateEdges]);

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
