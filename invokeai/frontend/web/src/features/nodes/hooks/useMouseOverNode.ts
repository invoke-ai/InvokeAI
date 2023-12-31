import { useStore } from '@nanostores/react';
import { atom } from 'nanostores';
import { useCallback, useMemo } from 'react';

export const $mouseOverNode = atom<string | null>(null);

export const useMouseOverNode = (nodeId: string) => {
  const mouseOverNode = useStore($mouseOverNode);

  const isMouseOverNode = useMemo(
    () => mouseOverNode === nodeId,
    [mouseOverNode, nodeId]
  );

  const handleMouseOver = useCallback(() => {
    $mouseOverNode.set(nodeId);
  }, [nodeId]);

  const handleMouseOut = useCallback(() => {
    $mouseOverNode.set(null);
  }, []);

  return { isMouseOverNode, handleMouseOver, handleMouseOut };
};
