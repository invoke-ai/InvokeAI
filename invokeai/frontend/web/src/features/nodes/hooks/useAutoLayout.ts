import { graphlib, layout } from '@dagrejs/dagre';
import type { Edge, NodeChange } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodesChanged } from 'features/nodes/store/nodesSlice';
import { selectEdges, selectNodes } from 'features/nodes/store/selectors';
import {
  selectLayeringStrategy,
  selectLayerSpacing,
  selectLayoutDirection,
  selectNodeAlignment,
  selectNodeSpacing,
} from 'features/nodes/store/workflowSettingsSlice';
import { NODE_WIDTH } from 'features/nodes/types/constants';
import type { AnyNode } from 'features/nodes/types/invocation';
import { isNotesNode } from 'features/nodes/types/invocation';
import { useCallback } from 'react';

const ESTIMATED_NOTES_NODE_HEIGHT = 200;
const DEFAULT_NODE_HEIGHT = NODE_WIDTH;

export const useAutoLayout = (): (() => void) => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector(selectNodes);
  const edges = useAppSelector(selectEdges);
  const nodeSpacing = useAppSelector(selectNodeSpacing);
  const layerSpacing = useAppSelector(selectLayerSpacing);
  const layeringStrategy = useAppSelector(selectLayeringStrategy);
  const layoutDirection = useAppSelector(selectLayoutDirection);
  const nodeAlignment = useAppSelector(selectNodeAlignment);

  const autoLayout = useCallback(() => {
    const g = new graphlib.Graph();

    g.setGraph({
      rankdir: layoutDirection,
      nodesep: nodeSpacing,
      ranksep: layerSpacing,
      ranker: layeringStrategy,
      align: nodeAlignment,
    });

    g.setDefaultEdgeLabel(() => ({}));

    const selectedNodes = nodes.filter((n) => n.selected);
    const isLayoutSelection = selectedNodes.length > 0 && nodes.length > selectedNodes.length;
    const nodesToLayout = isLayoutSelection ? selectedNodes : nodes;

    //Anchor of the selected nodes
    const selectionAnchor = {
      minX: Infinity,
      minY: Infinity,
    };

    nodesToLayout.forEach((node) => {
      // If we're laying out a selection, adjust the anchor to the top-left of the selection
      if (isLayoutSelection) {
        selectionAnchor.minX = Math.min(selectionAnchor.minX, node.position.x);
        selectionAnchor.minY = Math.min(selectionAnchor.minY, node.position.y);
      }
      // update the Height based on the node's measured height or use a default value
      const measuredHeight = node.measured?.height;
      const height =
        typeof measuredHeight === 'number'
          ? measuredHeight
          : isNotesNode(node)
            ? ESTIMATED_NOTES_NODE_HEIGHT
            : DEFAULT_NODE_HEIGHT;

      g.setNode(node.id, {
        width: node.width ?? NODE_WIDTH,
        height: height,
      });
    });

    let edgesToLayout: Edge[] = edges;
    if (isLayoutSelection) {
      const nodesToLayoutIds = new Set(nodesToLayout.map((n) => n.id));
      edgesToLayout = edges.filter((edge) => nodesToLayoutIds.has(edge.source) && nodesToLayoutIds.has(edge.target));
    }

    edgesToLayout.forEach((edge) => {
      g.setEdge(edge.source, edge.target);
    });

    layout(g);

    // anchor for the new layout
    const layoutAnchor = {
      minX: Infinity,
      minY: Infinity,
    };
    let offsetX = 0;
    let offsetY = 0;

    if (isLayoutSelection) {
      // Get the top-left position of the new layout
      nodesToLayout.forEach((node) => {
        const nodeInfo = g.node(node.id);
        // Convert from center to top-left
        const topLeftX = nodeInfo.x - nodeInfo.width / 2;
        const topLeftY = nodeInfo.y - nodeInfo.height / 2;
        // Use the top-left coordinates to find the bounding box
        layoutAnchor.minX = Math.min(layoutAnchor.minX, topLeftX);
        layoutAnchor.minY = Math.min(layoutAnchor.minY, topLeftY);
      });
      // Calculate the offset needed to move the new layout to the original position
      offsetX = selectionAnchor.minX - layoutAnchor.minX;
      offsetY = selectionAnchor.minY - layoutAnchor.minY;
    }

    // Create position changes for each node based on the new layout
    const positionChanges: NodeChange<AnyNode>[] = nodesToLayout.map((node) => {
      const nodeInfo = g.node(node.id);
      // Convert from center-based position to top-left-based position
      const x = nodeInfo.x - nodeInfo.width / 2;
      const y = nodeInfo.y - nodeInfo.height / 2;
      const newPosition = {
        x: isLayoutSelection ? x + offsetX : x,
        y: isLayoutSelection ? y + offsetY : y,
      };
      return { id: node.id, type: 'position', position: newPosition };
    });

    dispatch(nodesChanged(positionChanges));
  }, [dispatch, edges, nodes, nodeSpacing, layerSpacing, layeringStrategy, layoutDirection, nodeAlignment]);

  return autoLayout;
};
