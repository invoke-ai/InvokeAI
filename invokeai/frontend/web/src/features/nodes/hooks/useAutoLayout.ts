import { useStore } from '@nanostores/react';
import type { NodeChange } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { ELK as ELKType, ElkExtendedEdge, ElkNode } from 'elkjs';
import * as ElkModule from 'elkjs/lib/elk.bundled.js';
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

// This is a workaround for a common issue with how ELKjs is packaged. The bundled script doesn't have a
// clean ES module export, so we import the module namespace and then extract the constructor, which may
// be on the `default` property or be the module itself.
const ElkConstructor = ((ElkModule as unknown as { default: unknown }).default ?? ElkModule) as new (
  options?: Record<string, unknown>
) => ELKType;
const elk: ELKType = new ElkConstructor();

// These are estimates for node dimensions, used as a fallback when the node has not yet been rendered.
const ESTIMATED_NODE_HEADER_HEIGHT = 40;
const ESTIMATED_NODE_FOOTER_HEIGHT = 20;
const ESTIMATED_FIELD_HEIGHT = 36;
const ESTIMATED_NOTES_NODE_HEIGHT = 200;

export const useAutoLayout = (): (() => Promise<void>) => {
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

    // Get all node elements from the DOM at once for performance, then create a map for fast lookups.
    const nodeElements = document.querySelectorAll<HTMLDivElement>('.react-flow__node');
    const nodeElementMap = new Map<string, HTMLDivElement>();
    nodeElements.forEach((el) => {
      const id = el.dataset.id;
      if (id) {
        nodeElementMap.set(id, el);
      }
    });

    const elkNodes: ElkNode[] = nodesToLayout.map((node) => {
      // First, try to get the live height from the DOM element. This is the most accurate.
      let height = nodeElementMap.get(node.id)?.offsetHeight;

      // If the DOM element isn't available or its height is too small (e.g. not fully rendered),
      // fall back to the height from the node state.
      if (!height || height < ESTIMATED_NODE_HEADER_HEIGHT) {
        height = node.height;
      }

      // If we still don't have a valid height, estimate it based on the node's template.
      if (!height || height < ESTIMATED_NODE_HEADER_HEIGHT) {
        if (isInvocationNode(node)) {
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
          height = ESTIMATED_NOTES_NODE_HEIGHT;
        }
      }

      const elkNode: ElkNode = {
        id: node.id,
        width: node.width || NODE_WIDTH,
        // Final fallback to a default height if all else fails.
        height: height && height >= ESTIMATED_NODE_HEADER_HEIGHT ? height : 200,
      };

      // If we are layouting a selection, we must provide the positions of all unselected nodes to
      // the layout engine. This allows the engine to position the selected nodes relative to them.
      if (isLayoutingSelection && !node.selected) {
        elkNode.x = node.position.x;
        elkNode.y = node.position.y;
      }

      return elkNode;
    });

    const elkEdges: ElkExtendedEdge[] = edgesToLayout.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    }));

    const layoutOptions: ElkNode['layoutOptions'] = {
      'elk.algorithm': 'layered',
      'elk.spacing.nodeNode': String(nodeSpacing),
      'elk.direction': layoutDirection,
      'elk.layered.spacing.nodeNodeBetweenLayers': String(layerSpacing),
      'elk.spacing.edgeNode': '50',
      'elk.layered.nodePlacement.strategy': nodePlacementStrategy,
      'elk.layered.layering.strategy': layeringStrategy,
    };

    const graph: ElkNode = {
      id: 'root',
      layoutOptions,
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
