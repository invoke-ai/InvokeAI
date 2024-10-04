import type { Node } from 'reactflow';

export const findUnoccupiedPosition = (nodes: Node[], x: number, y: number) => {
  let newX = x;
  let newY = y;
  while (nodes.find((n) => n.position.x === newX && n.position.y === newY)) {
    newX = Math.floor(newX + 50);
    newY = Math.floor(newY + 50);
  }
  return { x: newX, y: newY };
};
