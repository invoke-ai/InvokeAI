import { useStore } from '@nanostores/react';
import type { NodeChange } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ELK, { type ElkEdge, type ElkNode } from 'elkjs/lib/elk.bundled.js';
import { $templates, nodesChanged } from 'features/nodes/store/nodesSlice';
import { selectEdges, selectNodes } from 'features/nodes/store/selectors';
import {
  selectLayeringStrategy,
  selectLayerSpacing,
  selectLayoutDirection,
  selectNodePlacementStrategy,
  selectNodeSpacing,
} from 'features/nodes/store/workflowSettingsSlice';
import { NODE_WIDTH } from 'features/nodes/types/constants';
import type { AnyNode } from 'features/nodes/types/invocation';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { useCallback } from 'react';

const elk = new ELK();

// These are estimates for node dimensions, used as a fallback when the node has not yet been rendered.
const ESTIMATED_NODE_HEADER_HEIGHT = 40;
const ESTIMATED_NODE_FOOTER_HEIGHT = 20;
const ESTIMATED_FIELD_HEIGHT = 36;
const ESTIMATED_NOTES_NODE_HEIGHT = 200;

export const useAutoLayout = () => {
  const dispatch = useAppDispatch();
  const nodes = useAppSelector(selectNodes);
  const edges = useAppSelector(selectEdges);
  const templates = useStore($templates);
  const nodePlacementStrategy = useAppSelector(selectNodePlacementStrategy);
  const layeringStrategy = useAppSelector(selectLayeringStrategy);
  const nodeSpacing = useAppSelector(selectNodeSpacing);
  const layerSpacing = useAppSelector(selectLayerSpacing);
  const layoutDirection = useAppSelector(selectLayoutDirection);

  const autoLayout = useCallback(async () => {
    const selectedNodes = nodes.filter((n) => n.selected);
    const isLayoutingSelection = selectedNodes.length > 0;

    // We always include all nodes in the layout, so the layout engine can avoid overlaps.
    const nodesToLayout = nodes;

    const nodeIdsToLayout = new Set(nodesToLayout.map((n) => n.id));
    const edgesToLayout = edges.filter((e) => nodeIdsToLayout.has(e.source) && nodeIdsToLayout.has(e.target));

    const elkNodes: ElkNode[] = nodesToLayout.map((node) => {
      let height = node.height;

      // If the node has no height, we need to estimate it.
      if (!height) {
        if (isInvocationNode(node)) {
          // This is an invocation node. We can estimate its height based on the number of fields.
          const template = templates[node.data.type];
          if (template) {
            const numInputs = Object.keys(template.inputs).length;
            const numOutputs = Object.keys(template.outputs).length;
            height =
              ESTIMATED_NODE_HEADER_HEIGHT +
              (numInputs + numOutputs) * ESTIMATED_FIELD_HEIGHT +
              ESTIMATED_NODE_FOOTER_HEIGHT;
          }
        } else if (node.type === 'notes') {
          // This is a notes node. They have a fixed default size.
          height = ESTIMATED_NOTES_NODE_HEIGHT;
        }
      }

      const elkNode: ElkNode = {
        id: node.id,
        width: node.width || NODE_WIDTH,
        height: height || 200, // A final fallback just in case.
      };

      // If we are layouting a selection, we must provide the positions of all unselected nodes to
      // the layout engine. This allows the engine to position the selected nodes relative to them.
      if (isLayoutingSelection && !node.selected) {
        elkNode.x = node.position.x;
        elkNode.y = node.position.y;
      }

      return elkNode;
    });

    const elkEdges: ElkEdge[] = edgesToLayout.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    }));

    const graph: ElkNode = {
      id: 'root',
      width: 0,
      height: 0,
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': layoutDirection,
        // Spacing between nodes in the same layer (vertical)
        'elk.spacing.nodeNode': String(nodeSpacing),
        // Spacing between nodes in adjacent layers (horizontal)
        'elk.layered.spacing.nodeNodeBetweenLayers': String(layerSpacing),
        // Spacing between an edge and a node
        'elk.spacing.edgeNode': '50',
        // layout strategy for node placement
        'elk.layered.nodePlacement.strategy': nodePlacementStrategy,
        // layering strategy
        'elk.layered.layering.strategy': layeringStrategy,
      },
      children: elkNodes,
      edges: elkEdges,
    };

    const layout = await elk.layout(graph);

    const positionChanges: NodeChange<AnyNode>[] =
      layout.children
        ?.filter((elkNode) => {
          // If we are layouting a selection, we only want to update the positions of the selected nodes.
          if (isLayoutingSelection) {
            return selectedNodes.some((n) => n.id === elkNode.id);
          }
          // Otherwise, update all nodes.
          return true;
        })
        .map((elkNode) => ({
          id: elkNode.id,
          type: 'position',
          position: { x: elkNode.x ?? 0, y: elkNode.y ?? 0 },
        })) ?? [];

    dispatch(nodesChanged(positionChanges));
  }, [
    nodes,
    edges,
    dispatch,
    templates,
    nodePlacementStrategy,
    layeringStrategy,
    nodeSpacing,
    layerSpacing,
    layoutDirection,
  ]);

  return autoLayout;
};
