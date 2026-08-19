import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';

import { Badge, Button, Stack, Text } from '@chakra-ui/react';
import { getTopologicalOrder } from '@features/workflow/core/graphLayout';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getNodeSubtitle } from './nodeSummaries';

type PreviewListNode = WorkflowPreviewGraph['nodes'][number];

const GraphPreviewListRow = ({ node, onSelect }: { node: PreviewListNode; onSelect: (nodeId: string) => void }) => {
  const { t } = useTranslation();
  const subtitle =
    getNodeSubtitle(node, t) ??
    `${node.id} · ${t('graphPreview.inputCount', { count: Object.keys(node.inputs).length })}`;
  const handleClick = useCallback(() => onSelect(node.id), [node.id, onSelect]);

  return (
    <Button
      fontWeight="normal"
      h="auto"
      justifyContent="flex-start"
      px="3"
      py="2"
      variant="ghost"
      w="full"
      onClick={handleClick}
    >
      <Stack align="flex-start" gap="0.5" w="full">
        <Badge fontFamily="mono" size="xs">
          {node.type}
        </Badge>
        <Text color="fg.muted" fontSize="2xs" fontWeight="normal" truncate>
          {subtitle}
        </Text>
      </Stack>
    </Button>
  );
};

/**
 * The preview dialog's "List" mode: every node as a full-width row, in
 * topological order, so a keyboard/screen-reader user can reach any node
 * without depending on the flow canvas. Selecting a row opens the same
 * inspector the flow's node click does (`GraphPreviewDialog.selectAndReveal`).
 *
 * Deliberately avoids `@platform/ui` — the barrel's fan-in budget is nearly
 * exhausted, and this component only needs plain Chakra primitives.
 */
export const GraphPreviewList = ({
  graph,
  onSelect,
}: {
  graph: WorkflowPreviewGraph;
  onSelect: (nodeId: string) => void;
}) => {
  // Live sources (e.g. Generate) recompile on every keystroke — re-sorting and
  // re-mapping the node list on every one of those renders (not just when the
  // graph itself changes) would make leaving List mode open needlessly costly.
  const orderedNodes = useMemo(() => {
    const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
    const order = getTopologicalOrder(
      graph.nodes,
      graph.edges.map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
    );

    return order.map((nodeId) => nodesById.get(nodeId)).filter((node): node is PreviewListNode => node !== undefined);
  }, [graph]);

  return (
    <Stack gap="1" h="full" overflowY="auto" p="2">
      {orderedNodes.map((node) => (
        <GraphPreviewListRow key={node.id} node={node} onSelect={onSelect} />
      ))}
    </Stack>
  );
};
