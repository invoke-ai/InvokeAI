import type { MutableRefObject } from 'react';
import { useCallback, useRef, useState } from 'react';

export const useMousePan = (ref: MutableRefObject<unknown>) => {
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 });
  const isPanningRef = useRef(false);
  const prevMousePositionRef = useRef({ x: 0, y: 0 });

  const handleMousePanDown = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault();
    isPanningRef.current = true;
    prevMousePositionRef.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handleMousePanUp = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault();
    isPanningRef.current = false;
  }, []);

  const handleMousePanMove = useCallback(
    (e: React.MouseEvent<HTMLImageElement>) => {
      if (isPanningRef.current && ref.current) {
        const deltaX = e.clientX - prevMousePositionRef.current.x;
        const deltaY = e.clientY - prevMousePositionRef.current.y;
        prevMousePositionRef.current = { x: e.clientX, y: e.clientY };
        setPanPosition((prevPosition) => ({
          x: prevPosition.x + deltaX,
          y: prevPosition.y + deltaY,
        }));
      }
    },
    [ref]
  );

  const resetPan = useCallback(() => {
    setPanPosition({ x: 0, y: 0 });
  }, []);

  return { panPosition, handleMousePanDown, handleMousePanUp, handleMousePanMove, resetPan };
};
