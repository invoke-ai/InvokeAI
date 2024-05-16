import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import { $copiedEdges,$copiedNodes,$cursorPos, selectionPasted, selectNodesSlice   } from 'features/nodes/store/nodesSlice';
import { findUnoccupiedPosition } from 'features/nodes/store/util/findUnoccupiedPosition';
import { v4 as uuidv4 } from 'uuid';

const copySelection = () => {
  // Use the imperative API here so we don't have to pass the whole slice around
  const { getState } = getStore();
  const { nodes, edges } = selectNodesSlice(getState());
  const selectedNodes = nodes.filter((node) => node.selected);
  const selectedEdges = edges.filter((edge) => edge.selected);
  $copiedNodes.set(selectedNodes);
  $copiedEdges.set(selectedEdges);
};

const pasteSelection = () => {
  const { getState, dispatch } = getStore();
  const currentNodes = selectNodesSlice(getState()).nodes;
  const cursorPos = $cursorPos.get();

  const copiedNodes = deepClone($copiedNodes.get());
  const copiedEdges = deepClone($copiedEdges.get());

  // Calculate an offset to reposition nodes to surround the cursor position, maintaining relative positioning
  const xCoords = copiedNodes.map((node) => node.position.x);
  const yCoords = copiedNodes.map((node) => node.position.y);
  const minX = Math.min(...xCoords);
  const minY = Math.min(...yCoords);
  const offsetX = cursorPos ? cursorPos.x - minX : 50;
  const offsetY = cursorPos ? cursorPos.y - minY : 50;

  copiedNodes.forEach((node) => {
    const { x, y } = findUnoccupiedPosition(currentNodes, node.position.x + offsetX, node.position.y + offsetY);
    node.position.x = x;
    node.position.y = y;
    // Pasted nodes are selected
    node.selected = true;
    // Also give em a fresh id
    const id = uuidv4();
    // Update the edges to point to the new node id
    for (const edge of copiedEdges) {
      if (edge.source === node.id) {
        edge.source = id;
        edge.id = edge.id.replace(node.data.id, id);
      }
      if (edge.target === node.id) {
        edge.target = id;
        edge.id = edge.id.replace(node.data.id, id);
      }
    }
    node.id = id;
    node.data.id = id;
  });

  dispatch(selectionPasted({ nodes: copiedNodes, edges: copiedEdges }));
};

const api = { copySelection, pasteSelection };

export const useCopyPaste = () => {
  return api;
};
