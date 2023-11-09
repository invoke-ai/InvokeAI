import { Node } from 'reactflow';

export const findUnoccupiedPosition = (nodes: Node[], x: number, y: number) => {
  let newX = x;
  let newY = y;
  while (nodes.find((n) => n.position.x === newX && n.position.y === newY)) {
    newX = newX + 50;
    newY = newY + 50;
  }
  return { x: newX, y: newY };
};
