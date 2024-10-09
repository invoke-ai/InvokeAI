import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import {
  $copiedEdges,
  $copiedNodes,
  $cursorPos,
  $edgesToCopiedNodes,
  $templates,
  edgesChanged,
  nodesChanged,
} from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import { findUnoccupiedPosition } from 'features/nodes/store/util/findUnoccupiedPosition';
import { validateConnection } from 'features/nodes/store/util/validateConnection';
import { t } from 'i18next';
import { isEqual, isNil, uniqWith } from 'lodash-es';
import type { EdgeChange, NodeChange } from 'reactflow';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

const log = logger('workflows');

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

const _pasteSelection = (withEdgesToCopiedNodes?: boolean) => {
  const { getState, dispatch } = getStore();
  const { nodes, edges } = selectNodesSlice(getState());
  const templates = $templates.get();
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

  // When we validate the new edges, we need to include the copied nodes as well as the existing nodes,
  // else the edges will all fail bc they point to nodes that don't exist yet
  const validationNodes = [...nodes, ...copiedNodes];
  // As an edge is validated, we will need to add it to the list of edges used for validation, because
  // validation may depend on the existence of other edges
  const validationEdges = [...edges];

  // Add new edges
  copiedEdges.forEach((e) => {
    const { source, sourceHandle, target, targetHandle } = e;
    // We need a type guard here to work around reactflow types
    assert(!isNil(sourceHandle));
    assert(!isNil(targetHandle));

    // Validate the edge before adding it
    const validationResult = validateConnection(
      { source, sourceHandle, target, targetHandle },
      validationNodes,
      validationEdges,
      templates,
      null,
      true
    );
    // If the edge is invalid, log a warning and skip it
    if (!validationResult.isValid) {
      log.warn(
        { edge: { source, sourceHandle, target, targetHandle } },
        `Invalid edge, cannot paste: ${t(validationResult.messageTKey)}`
      );
      return;
    }
    edgeChanges.push({
      type: 'add',
      item: e,
    });
    // Add the edge to the list of edges used for validation so that subsequent edges can depend on it
    validationEdges.push(e);
  });
  if (nodeChanges.length > 0) {
    dispatch(nodesChanged(nodeChanges));
  }
  if (edgeChanges.length > 0) {
    dispatch(edgesChanged(edgeChanges));
  }
};

const pasteSelection = () => {
  _pasteSelection(false);
};

const pasteSelectionWithEdges = () => {
  _pasteSelection(true);
};

const api = { copySelection, pasteSelection, pasteSelectionWithEdges };

export const useCopyPaste = () => {
  return api;
};
