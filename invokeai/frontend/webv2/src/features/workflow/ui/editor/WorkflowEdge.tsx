import type { EdgeProps } from '@xyflow/react';

import { getBezierPath, getSmoothStepPath } from '@xyflow/react';
import { memo, type CSSProperties } from 'react';
import { useTranslation } from 'react-i18next';

import type { WorkflowEdgeData } from './flowAdapters';

const FALLBACK_EDGE_DATA: WorkflowEdgeData = {
  fieldTypeLabel: null,
  pathType: 'default',
  stroke: 'var(--xy-edge-stroke)',
  strokeWidth: 2,
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
  const { t } = useTranslation();
  const edgeData = (data as WorkflowEdgeData | undefined) ?? FALLBACK_EDGE_DATA;
  const collapsedCount = edgeData.collapsedEdgeIds?.length ?? 0;
  // One drawn edge for several fields carries no single type: it draws neutral, as legacy's did.
  const isMergedEdge = collapsedCount > 1;
  // Worded as the handle tooltips word the same type.
  const tooltip = edgeData.isLoopLinkage
    ? t('nodes.loopLinkage')
    : isMergedEdge
      ? t('nodes.collapsedConnections', { count: collapsedCount })
      : edgeData.fieldTypeLabel === null
        ? t('nodes.unknownFieldType')
        : edgeData.isBatch
          ? `${edgeData.fieldTypeLabel} ${t('nodes.batch')}`
          : edgeData.fieldTypeLabel;
  const [edgePath, labelX, labelY] =
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
    stroke: isMergedEdge ? FALLBACK_EDGE_DATA.stroke : edgeData.stroke,
    strokeDasharray: selected ? '5' : isMergedEdge ? undefined : edgeData.strokeDasharray,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    strokeWidth: isMergedEdge ? FALLBACK_EDGE_DATA.strokeWidth : edgeData.strokeWidth,
  };

  return (
    <g aria-label={tooltip} role="img">
      <path
        aria-label={tooltip}
        className="react-flow__edge-path"
        d={edgePath}
        fill="none"
        markerEnd={markerEnd}
        markerStart={markerStart}
        style={edgeStyle}
      >
        <title>{tooltip}</title>
      </path>
      {interactionWidth ? (
        <path
          aria-label={tooltip}
          className="react-flow__edge-interaction"
          d={edgePath}
          fill="none"
          strokeOpacity={0}
          strokeWidth={interactionWidth}
        >
          <title>{tooltip}</title>
        </path>
      ) : null}
      {collapsedCount > 0 ? <CollapsedEdgeCount count={collapsedCount} x={labelX} y={labelY} /> : null}
    </g>
  );
};

const BADGE_HEIGHT = 16;

/** Legacy's count on an edge between two collapsed nodes: how many connections it stands for. */
const CollapsedEdgeCount = ({ count, x, y }: { count: number; x: number; y: number }) => {
  const width = Math.max(BADGE_HEIGHT, 8 + String(count).length * 7);

  return (
    <g aria-hidden data-collapsed-edge-count={count} pointerEvents="none" transform={`translate(${x} ${y})`}>
      <rect
        fill="var(--wb-node-surface)"
        height={BADGE_HEIGHT}
        rx={BADGE_HEIGHT / 2}
        stroke="var(--xy-edge-stroke)"
        width={width}
        x={-width / 2}
        y={-BADGE_HEIGHT / 2}
      />
      <text
        dominantBaseline="central"
        fill="var(--wb-edge-badge-fg)"
        fontSize="10.5"
        fontWeight="600"
        textAnchor="middle"
      >
        {count}
      </text>
    </g>
  );
};

export const WorkflowEdge = memo(WorkflowEdgeComponent);
