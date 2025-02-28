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

const $mouseOverFormField = atom<string | null>(null);

export const useMouseOverFormField = (nodeId: string) => {
  const [isMouseOverFormField, setIsMouseOverFormField] = useState(false);

  useEffect(() => {
    const unsubscribe = $mouseOverFormField.subscribe((v) => {
      setIsMouseOverFormField(v === nodeId);
    });
    return unsubscribe;
  }, [isMouseOverFormField, nodeId]);

  const handleMouseOver = useCallback(() => {
    $mouseOverFormField.set(nodeId);
  }, [nodeId]);

  const handleMouseOut = useCallback(() => {
    $mouseOverFormField.set(null);
  }, []);

  return { isMouseOverFormField, handleMouseOver, handleMouseOut };
};
