import type { EdgeProps } from '@xyflow/react';

import { getBezierPath, getSmoothStepPath } from '@xyflow/react';
import { memo, type CSSProperties } from 'react';

import type { WorkflowEdgeData } from './flowAdapters';

const FALLBACK_EDGE_DATA: WorkflowEdgeData = {
  fieldTypeLabel: 'Unknown',
  pathType: 'default',
  stroke: 'var(--xy-edge-stroke)',
  strokeWidth: 2,
  tooltip: 'Unknown field type',
};

const WorkflowEdgeComponent = ({
  data,
  interactionWidth = 20,
  markerEnd,
  markerStart,
  selected,
  sourcePosition,
  sourceX,
  sourceY,
  style,
  targetPosition,
  targetX,
  targetY,
}: EdgeProps) => {
  const edgeData = (data as WorkflowEdgeData | undefined) ?? FALLBACK_EDGE_DATA;
  const [edgePath] =
    edgeData.pathType === 'step'
      ? getSmoothStepPath({
          borderRadius: 0,
          sourcePosition,
          sourceX,
          sourceY,
          targetPosition,
          targetX,
          targetY,
        })
      : getBezierPath({
          sourcePosition,
          sourceX,
          sourceY,
          targetPosition,
          targetX,
          targetY,
        });
  const edgeStyle: CSSProperties = {
    ...style,
    animation: selected
      ? 'dashdraw var(--wb-motion-duration-slow) linear var(--wb-motion-animation-iteration-count)'
      : undefined,
    stroke: edgeData.stroke,
    strokeDasharray: selected ? '5' : edgeData.strokeDasharray,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    strokeWidth: edgeData.strokeWidth,
  };

  return (
    <g aria-label={edgeData.tooltip} role="img">
      <path
        aria-label={edgeData.tooltip}
        className="react-flow__edge-path"
        d={edgePath}
        fill="none"
        markerEnd={markerEnd}
        markerStart={markerStart}
        style={edgeStyle}
      >
        <title>{edgeData.tooltip}</title>
      </path>
      {interactionWidth ? (
        <path
          aria-label={edgeData.tooltip}
          className="react-flow__edge-interaction"
          d={edgePath}
          fill="none"
          strokeOpacity={0}
          strokeWidth={interactionWidth}
        >
          <title>{edgeData.tooltip}</title>
        </path>
      ) : null}
    </g>
  );
};

export const WorkflowEdge = memo(WorkflowEdgeComponent);
