import type { XYPosition } from '@workbench/workflows/types';

import { Box } from '@chakra-ui/react';
import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';

import type { WorkflowFlowInstance } from './flowInstanceStore';

/**
 * Freeform lasso selection for the flow editor. xyflow has no core lasso (its
 * official one is a copy-in component), so this follows the same approach:
 * capture-phase pointer handlers on the flow wrapper stop the pane's own drag,
 * the path is hit-tested against node bounds in flow coordinates, and a
 * pointer-events-free SVG overlay draws the path. Partial mode: a node is
 * selected when any corner or its center falls inside the polygon.
 */

const MIN_POLYGON_POINTS = 3;

/** Ray-casting point-in-polygon. */
const isInsidePolygon = (point: XYPosition, polygon: XYPosition[]): boolean => {
  let isInside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i] as XYPosition;
    const b = polygon[j] as XYPosition;

    if (a.y > point.y !== b.y > point.y && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) {
      isInside = !isInside;
    }
  }

  return isInside;
};

export const useLasso = ({
  enabled,
  flowInstance,
  onSelect,
}: {
  enabled: boolean;
  flowInstance: WorkflowFlowInstance | null;
  onSelect: (nodeIds: string[]) => void;
}): {
  /** Spread onto the flow wrapper (capture phase, so the pane never sees the drag). */
  lassoHandlers: {
    onClickCapture: (event: ReactMouseEvent<HTMLDivElement>) => void;
    onPointerDownCapture: (event: ReactPointerEvent<HTMLDivElement>) => void;
  };
  /** Render inside the (position: relative) flow wrapper. */
  lassoOverlay: ReactNode;
} => {
  const [screenPoints, setScreenPoints] = useState<XYPosition[]>([]);
  const flowPointsRef = useRef<XYPosition[]>([]);
  const frameRef = useRef<number | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const suppressClickRef = useRef(false);

  useEffect(
    () => () => {
      cleanupRef.current?.();

      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
      }
    },
    []
  );

  const applySelection = () => {
    const polygon = flowPointsRef.current;

    if (!flowInstance || polygon.length < MIN_POLYGON_POINTS) {
      return;
    }

    const selectedIds: string[] = [];

    for (const node of flowInstance.getNodes()) {
      const width = node.measured?.width ?? 0;
      const height = node.measured?.height ?? 0;
      const { x, y } = node.position;
      const probes: XYPosition[] = [
        { x, y },
        { x: x + width, y },
        { x, y: y + height },
        { x: x + width, y: y + height },
        { x: x + width / 2, y: y + height / 2 },
      ];

      if (probes.some((probe) => isInsidePolygon(probe, polygon))) {
        selectedIds.push(node.id);
      }
    }

    onSelect(selectedIds);
  };

  const onPointerDownCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!enabled || !flowInstance || event.button !== 0) {
      return;
    }

    // Let clicks inside nodes/controls behave normally; the lasso only owns
    // drags that start on the pane itself.
    if (!(event.target instanceof Element) || !event.target.classList.contains('react-flow__pane')) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const container = event.currentTarget;
    const bounds = container.getBoundingClientRect();
    const toLocal = (clientX: number, clientY: number): XYPosition => ({
      x: clientX - bounds.left,
      y: clientY - bounds.top,
    });

    flowPointsRef.current = [flowInstance.screenToFlowPosition({ x: event.clientX, y: event.clientY })];
    setScreenPoints([toLocal(event.clientX, event.clientY)]);

    const onPointerMove = (moveEvent: globalThis.PointerEvent) => {
      flowPointsRef.current = [
        ...flowPointsRef.current,
        flowInstance.screenToFlowPosition({ x: moveEvent.clientX, y: moveEvent.clientY }),
      ];
      setScreenPoints((current) => [...current, toLocal(moveEvent.clientX, moveEvent.clientY)]);

      if (frameRef.current === null) {
        frameRef.current = requestAnimationFrame(() => {
          frameRef.current = null;
          applySelection();
        });
      }
    };

    const onPointerUp = () => {
      suppressClickRef.current = flowPointsRef.current.length >= MIN_POLYGON_POINTS;
      cleanupRef.current?.();
      applySelection();
      flowPointsRef.current = [];
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

  const lassoOverlay =
    screenPoints.length >= 2 ? (
      <Box as="svg" h="full" inset="0" overflow="visible" pointerEvents="none" position="absolute" w="full" zIndex="4">
        <polygon
          fill="var(--xy-selection-background-color)"
          points={screenPoints.map((point) => `${point.x},${point.y}`).join(' ')}
          stroke="var(--xy-edge-stroke-selected)"
          strokeDasharray="4 2"
          strokeWidth="1"
        />
      </Box>
    ) : null;

  return { lassoHandlers: { onClickCapture, onPointerDownCapture }, lassoOverlay };
};
