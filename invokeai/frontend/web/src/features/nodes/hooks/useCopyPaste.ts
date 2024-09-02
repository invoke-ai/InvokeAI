import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import {
  $copiedEdges,
  $copiedNodes,
  $cursorPos,
  $edgesToCopiedNodes,
  edgesChanged,
  nodesChanged,
  selectNodesSlice,
} from 'features/nodes/store/nodesSlice';
import { findUnoccupiedPosition } from 'features/nodes/store/util/findUnoccupiedPosition';
import { isEqual, uniqWith } from 'lodash-es';
import type { EdgeChange, NodeChange } from 'reactflow';
import { v4 as uuidv4 } from 'uuid';

const copySelection = () => {
  // Use the imperative API here so we don't have to pass the whole slice around
  const { getState } = getStore();
  const { nodes, edges } = selectNodesSlice(getState());
  const selectedNodes = nodes.filter((node) => node.selected);
  const selectedEdges = edges.filter((edge) => edge.selected);
  const edgesToSelectedNodes = edges.filter((edge) => selectedNodes.some((node) => node.id === edge.target));
  $copiedNodes.set(selectedNodes);
  $copiedEdges.set(selectedEdges);
  $edgesToCopiedNodes.set(edgesToSelectedNodes);
};

const pasteSelection = (withEdgesToCopiedNodes?: boolean) => {
  const { getState, dispatch } = getStore();
  const { nodes, edges } = selectNodesSlice(getState());
  const cursorPos = $cursorPos.get();

  const copiedNodes = deepClone($copiedNodes.get());
  let copiedEdges = deepClone($copiedEdges.get());

  if (withEdgesToCopiedNodes) {
    const edgesToCopiedNodes = deepClone($edgesToCopiedNodes.get());
    copiedEdges = uniqWith([...copiedEdges, ...edgesToCopiedNodes], isEqual);
  }

  // Calculate an offset to reposition nodes to surround the cursor position, maintaining relative positioning
  const xCoords = copiedNodes.map((node) => node.position.x);
  const yCoords = copiedNodes.map((node) => node.position.y);
  const minX = Math.min(...xCoords);
  const minY = Math.min(...yCoords);
  const offsetX = cursorPos ? cursorPos.x - minX : 50;
  const offsetY = cursorPos ? cursorPos.y - minY : 50;

  copiedNodes.forEach((node) => {
    const { x, y } = findUnoccupiedPosition(nodes, node.position.x + offsetX, node.position.y + offsetY);
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
      } else if (edge.target === node.id) {
        edge.target = id;
      }
    }
    node.id = id;
    node.data.id = id;
  });

  copiedEdges.forEach((edge) => {
    // Copied edges need a fresh id too
    edge.id = uuidv4();
  });

  const nodeChanges: NodeChange[] = [];
  const edgeChanges: EdgeChange[] = [];
  // Deselect existing nodes
  nodes.forEach(({ id, selected }) => {
    if (selected) {
      nodeChanges.push({
        type: 'select',
        id,
        selected: false,
      });
    }
  });
  // Add new nodes
  copiedNodes.forEach((n) => {
    nodeChanges.push({
      type: 'add',
      item: n,
    });
  });
  // Deselect existing edges
  edges.forEach(({ id, selected }) => {
    if (selected) {
      edgeChanges.push({
        type: 'select',
        id,
        selected: false,
      });
    }
  });
  // Add new edges
  copiedEdges.forEach((e) => {
    edgeChanges.push({
      type: 'add',
      item: e,
    });
  });
  if (nodeChanges.length > 0) {
    dispatch(nodesChanged(nodeChanges));
  }
  if (edgeChanges.length > 0) {
    dispatch(edgesChanged(edgeChanges));
  }
};

const api = { copySelection, pasteSelection };

export const useCopyPaste = () => {
  return api;
};
