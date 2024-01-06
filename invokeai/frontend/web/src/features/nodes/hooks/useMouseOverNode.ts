import { atom } from 'nanostores';
import { useCallback, useEffect, useState } from 'react';

export const $mouseOverNode = atom<string | null>(null);

export const useMouseOverNode = (nodeId: string) => {
  const [isMouseOverNode, setIsMouseOverNode] = useState(false);

  useEffect(() => {
    const unsubscribe = $mouseOverNode.subscribe((v) => {
      setIsMouseOverNode(v === nodeId);
    });
    return unsubscribe;
  }, [isMouseOverNode, nodeId]);

  const handleMouseOver = useCallback(() => {
    $mouseOverNode.set(nodeId);
  }, [nodeId]);

  const handleMouseOut = useCallback(() => {
    $mouseOverNode.set(null);
  }, []);

  return { isMouseOverNode, handleMouseOver, handleMouseOut };
};
