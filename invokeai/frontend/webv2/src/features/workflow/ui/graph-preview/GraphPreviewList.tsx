import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type { ListRowProps } from '@platform/ui/list/List';

import { getTopologicalOrder } from '@features/workflow/core/graphLayout';
import { List } from '@platform/ui/list/List';
import { ListItem } from '@platform/ui/list/ListItem';
import { listRowsFromItems } from '@platform/ui/list/listRows';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getNodeSubtitle } from './nodeSummaries';

type PreviewListNode = WorkflowPreviewGraph['nodes'][number];

const getNodeId = (node: PreviewListNode): string => node.id;

const GraphPreviewListRow = memo(function GraphPreviewListRow({
  node,
  onSelect,
  ...rowProps
}: ListRowProps & { node: PreviewListNode; onSelect: (nodeId: string) => void }) {
  const { t } = useTranslation();
  const subtitle =
    getNodeSubtitle(node, t) ??
    `${node.id} · ${t('graphPreview.inputCount', { count: Object.keys(node.inputs).length })}`;
  const handlePress = useCallback(() => onSelect(node.id), [node.id, onSelect]);

  return <ListItem {...rowProps} description={subtitle} title={node.type} onPress={handlePress} />;
});

/** Every node in topological order, for keyboard and screen-reader access to the shared inspector. */
export const GraphPreviewList = ({
  graph,
  selectedNodeId,
  onSelect,
}: {
  graph: WorkflowPreviewGraph;
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
}) => {
  const { t } = useTranslation();
  // Recompute sorted rows only when the graph changes, not on unrelated live-source renders.
  const rows = useMemo(() => {
    const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
    const order = getTopologicalOrder(
      graph.nodes,
      graph.edges.map((edge) => ({ sourceNodeId: edge.sourceNodeId, targetNodeId: edge.targetNodeId }))
    );
    const orderedNodes = order
      .map((nodeId) => nodesById.get(nodeId))
      .filter((node): node is PreviewListNode => node !== undefined);

    return listRowsFromItems(orderedNodes, getNodeId);
  }, [graph]);
  const renderItem = useCallback(
    (node: PreviewListNode, rowProps: ListRowProps) => (
      <GraphPreviewListRow {...rowProps} node={node} onSelect={onSelect} />
    ),
    [onSelect]
  );

  return (
    <List
      activeKey={selectedNodeId}
      dividers
      label={t('graphPreview.nodes')}
      renderItem={renderItem}
      revealActiveOnMount
      rows={rows}
    />
  );
};
