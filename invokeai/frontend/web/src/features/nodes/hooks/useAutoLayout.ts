import { graphlib, layout } from '@dagrejs/dagre';
import type { Edge, NodeChange} from '@xyflow/react';
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
const NODE_PADDING = 0;//40; // Padding to add to the node height

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

    // Get the top-left position of the selection's bounding box before layout
    const selectionBBox = {
      minX: Infinity,
      minY: Infinity,
    };

    if (isLayoutSelection) {
      for (const node of selectedNodes) {
        selectionBBox.minX = Math.min(selectionBBox.minX, node.position.x);
        selectionBBox.minY = Math.min(selectionBBox.minY, node.position.y);
      }
    }

    nodesToLayout.forEach((node) => {
      let height: number;

      // Check if a measured height is available and valid
      if (node.measured?.height !== null && node.measured?.height !== undefined) {
        height = node.measured.height + NODE_PADDING; // Add padding to the measured height
      } else {
        // If not available, determine the fallback height
        height = isNotesNode(node) ? ESTIMATED_NOTES_NODE_HEIGHT : DEFAULT_NODE_HEIGHT;
      }

      g.setNode(node.id, {
        width: node.width ?? NODE_WIDTH,
        height: height,
      });
    });

    const edgesToLayout: Edge[] = isLayoutSelection
      ? edges.filter(
          (edge) =>
            nodesToLayout.some((n) => n.id === edge.source) && nodesToLayout.some((n) => n.id === edge.target)
        )
      : edges;

    edgesToLayout.forEach((edge) => {
      g.setEdge(edge.source, edge.target);
    });

    layout(g);

    const layoutBBox = {
      minX: Infinity,
      minY: Infinity,
    };
    let offsetX = 0;
    let offsetY = 0;

    if (isLayoutSelection) {
      // Get the top-left position of the new layout's bounding box
      nodesToLayout.forEach((node) => {
        const { x, y } = g.node(node.id);
        layoutBBox.minX = Math.min(layoutBBox.minX, x);
        layoutBBox.minY = Math.min(layoutBBox.minY, y);
      });

      // Calculate the offset needed to move the new layout to the original position
      offsetX = selectionBBox.minX - layoutBBox.minX;
      offsetY = selectionBBox.minY - layoutBBox.minY;
    }

    const positionChanges: NodeChange<AnyNode>[] = nodesToLayout.map((node) => {
      const { x, y } = g.node(node.id);
      // For selected layouts, apply the calculated offset. Otherwise, use the new position directly.
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
