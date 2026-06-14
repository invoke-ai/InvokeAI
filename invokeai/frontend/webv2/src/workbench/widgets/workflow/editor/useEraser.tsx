import { Box } from '@chakra-ui/react';
import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';

import type { XYPosition } from '../../../workflows/types';
import type { WorkflowFlowInstance } from './flowInstanceStore';

const ERASER_RADIUS_PX = 14;
const EDGE_SAMPLE_DISTANCE = 12;

type ErasedElements = { edgeIds: string[]; nodeIds: string[] };

const getDistanceToSegmentSquared = (point: XYPosition, start: XYPosition, end: XYPosition): number => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;

  if (lengthSquared === 0) {
    return (point.x - start.x) ** 2 + (point.y - start.y) ** 2;
  }

  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  const closest = { x: start.x + t * dx, y: start.y + t * dy };

  return (point.x - closest.x) ** 2 + (point.y - closest.y) ** 2;
};

const getEraserRect = (start: XYPosition, end: XYPosition, radius: number) => ({
  height: Math.abs(end.y - start.y) + radius * 2,
  width: Math.abs(end.x - start.x) + radius * 2,
  x: Math.min(start.x, end.x) - radius,
  y: Math.min(start.y, end.y) - radius,
});

const getLocalPoint = (bounds: DOMRect, clientX: number, clientY: number): XYPosition => ({
  x: clientX - bounds.left,
  y: clientY - bounds.top,
});

const isEraserTarget = (target: EventTarget | null): target is Element =>
  target instanceof Element && target.closest('.react-flow__pane, .react-flow__node, .react-flow__edge') !== null;

const getIntersectingEdgeIds = ({
  container,
  erasedEdgeIds,
  end,
  radius,
  start,
}: {
  container: HTMLElement;
  erasedEdgeIds: Set<string>;
  end: XYPosition;
  radius: number;
  start: XYPosition;
}): string[] => {
  const edgeIds: string[] = [];
  const radiusSquared = radius * radius;
  const sampleDistance = Math.max(radius, EDGE_SAMPLE_DISTANCE);

  for (const path of container.querySelectorAll<SVGPathElement>('.react-flow__edge-path')) {
    const edgeId = path.closest<SVGGElement>('.react-flow__edge')?.dataset.id;

    if (!edgeId || erasedEdgeIds.has(edgeId)) {
      continue;
    }

    let length: number;

    try {
      length = path.getTotalLength();
    } catch {
      continue;
    }

    for (let distance = 0; distance <= length; distance += sampleDistance) {
      const point = path.getPointAtLength(distance);

      if (getDistanceToSegmentSquared(point, start, end) <= radiusSquared) {
        edgeIds.push(edgeId);
        break;
      }
    }

    if (!edgeIds.includes(edgeId)) {
      const point = path.getPointAtLength(length);

      if (getDistanceToSegmentSquared(point, start, end) <= radiusSquared) {
        edgeIds.push(edgeId);
      }
    }
  }

  return edgeIds;
};

export const useEraser = ({
  enabled,
  flowInstance,
  onErase,
}: {
  enabled: boolean;
  flowInstance: WorkflowFlowInstance | null;
  onErase: (elements: ErasedElements) => void;
}): {
  eraserHandlers: {
    onClickCapture: (event: ReactMouseEvent<HTMLDivElement>) => void;
    onPointerDownCapture: (event: ReactPointerEvent<HTMLDivElement>) => void;
  };
  eraserOverlay: ReactNode;
} => {
  const [screenPoints, setScreenPoints] = useState<XYPosition[]>([]);
  const boundsRef = useRef<DOMRect | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const erasedEdgeIdsRef = useRef(new Set<string>());
  const erasedNodeIdsRef = useRef(new Set<string>());
  const flowPointsRef = useRef<XYPosition[]>([]);
  const suppressClickRef = useRef(false);

  useEffect(
    () => () => {
      cleanupRef.current?.();
    },
    []
  );

  const eraseSegment = (container: HTMLElement, start: XYPosition, end: XYPosition) => {
    if (!flowInstance) {
      return;
    }

    const radius = ERASER_RADIUS_PX / flowInstance.getZoom();
    const nodeIds = flowInstance
      .getIntersectingNodes(getEraserRect(start, end, radius), true)
      .map((node) => node.id)
      .filter((nodeId) => !erasedNodeIdsRef.current.has(nodeId));
    const nodeIdSet = new Set(nodeIds);

    for (const edge of flowInstance.getEdges()) {
      if (nodeIdSet.has(edge.source) || nodeIdSet.has(edge.target)) {
        erasedEdgeIdsRef.current.add(edge.id);
      }
    }

    const edgeIds = getIntersectingEdgeIds({
      container,
      erasedEdgeIds: erasedEdgeIdsRef.current,
      end,
      radius,
      start,
    });

    if (nodeIds.length === 0 && edgeIds.length === 0) {
      return;
    }

    for (const nodeId of nodeIds) {
      erasedNodeIdsRef.current.add(nodeId);
    }

    for (const edgeId of edgeIds) {
      erasedEdgeIdsRef.current.add(edgeId);
    }

    onErase({ edgeIds, nodeIds });
  };

  const addPoint = (container: HTMLElement, event: Pick<PointerEvent, 'clientX' | 'clientY'>) => {
    const bounds = boundsRef.current;

    if (!flowInstance || !bounds) {
      return;
    }

    const flowPoint = flowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY });
    const previousPoint = flowPointsRef.current.at(-1) ?? flowPoint;

    flowPointsRef.current = [...flowPointsRef.current, flowPoint];
    setScreenPoints((current) => [...current, getLocalPoint(bounds, event.clientX, event.clientY)]);
    eraseSegment(container, previousPoint, flowPoint);
  };

  const onPointerDownCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!enabled || !flowInstance || event.button !== 0 || !isEraserTarget(event.target)) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    suppressClickRef.current = true;
    erasedEdgeIdsRef.current = new Set();
    erasedNodeIdsRef.current = new Set();
    flowPointsRef.current = [];

    const container = event.currentTarget;
    boundsRef.current = container.getBoundingClientRect();
    setScreenPoints([]);
    addPoint(container, event.nativeEvent);

    const onPointerMove = (moveEvent: globalThis.PointerEvent) => {
      moveEvent.preventDefault();
      addPoint(container, moveEvent);
    };

    const onPointerUp = () => {
      cleanupRef.current?.();
      flowPointsRef.current = [];
      boundsRef.current = null;
      setScreenPoints([]);
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    cleanupRef.current = () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      cleanupRef.current = null;
    };
  };

  const onClickCapture = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!suppressClickRef.current) {
      return;
    }

    suppressClickRef.current = false;
    event.preventDefault();
    event.stopPropagation();
  };

  const points = screenPoints.map((point) => `${point.x},${point.y}`).join(' ');
  const eraserOverlay =
    screenPoints.length > 0 ? (
      <Box as="svg" h="full" inset="0" overflow="visible" pointerEvents="none" position="absolute" w="full" zIndex="4">
        <polyline
          fill="none"
          points={points}
          stroke="var(--xy-selection-background-color)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={ERASER_RADIUS_PX * 2}
        />
        <polyline
          fill="none"
          points={points}
          stroke="var(--xy-edge-stroke-selected)"
          strokeDasharray="4 3"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.5"
        />
      </Box>
    ) : null;

  return { eraserHandlers: { onClickCapture, onPointerDownCapture }, eraserOverlay };
};
