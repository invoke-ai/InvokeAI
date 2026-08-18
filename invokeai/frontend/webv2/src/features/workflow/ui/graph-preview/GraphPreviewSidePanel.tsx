import type { GraphPreviewSourceState, WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type { TFunction } from 'i18next';

import { Badge, Box, DataList, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton, Scrollable } from '@platform/ui';
import { ArrowLeftIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

type PreviewGraphNode = WorkflowPreviewGraph['nodes'][number];

/**
 * The preview dialog's right-hand rail: either a summary of the compiled
 * graph (node count, result destination, source-specific settings rows), or
 * — once a node is selected in the flow/list — that node's inspector (who
 * set its inputs, their resolved values, and its edges). The outer
 * `Scrollable` stays mounted across the switch so selecting a node doesn't
 * remount the panel's scroll container.
 */
export const GraphPreviewSidePanel = ({
  source,
  selectedNode,
  onBack,
  onProvenanceClick,
}: {
  source: GraphPreviewSourceState;
  selectedNode: PreviewGraphNode | null;
  onBack: () => void;
  onProvenanceClick: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <Scrollable flexShrink={0} h="full" label={t('graphPreview.thisGraph')} w="19rem">
      {selectedNode ? (
        <NodeInspector node={selectedNode} source={source} onBack={onBack} onProvenanceClick={onProvenanceClick} />
      ) : (
        <GraphSummary source={source} />
      )}
    </Scrollable>
  );
};

const GraphSummary = ({ source }: { source: GraphPreviewSourceState }) => {
  const { t } = useTranslation();
  const nodeCount = source.graph?.nodes.length;

  return (
    <Stack gap="2" p="2">
      <Stack gap="0.5">
        <Text fontSize="sm" fontWeight="semibold">
          {t('graphPreview.thisGraph')}
        </Text>
        <Text color="fg.muted" fontSize="2xs">
          {t('graphPreview.selectNode')}
        </Text>
      </Stack>
      <DataList.Root gap="1.5" orientation="horizontal" size="sm">
        <DataList.Item>
          <DataList.ItemLabel fontSize="2xs">{t('graphPreview.nodes')}</DataList.ItemLabel>
          <DataList.ItemValue fontSize="2xs" minW="0">
            {nodeCount !== undefined ? String(nodeCount) : '—'}
          </DataList.ItemValue>
        </DataList.Item>
        <DataList.Item>
          <DataList.ItemLabel fontSize="2xs">{t('graphPreview.destination')}</DataList.ItemLabel>
          <DataList.ItemValue fontSize="2xs" minW="0">
            {source.destinationLabel ?? '—'}
          </DataList.ItemValue>
        </DataList.Item>
        {source.summaryRows.map((row) => (
          <DataList.Item key={row.id}>
            <DataList.ItemLabel fontSize="2xs">{row.label}</DataList.ItemLabel>
            <DataList.ItemValue fontSize="2xs" minW="0">
              {row.value}
            </DataList.ItemValue>
          </DataList.Item>
        ))}
      </DataList.Root>
    </Stack>
  );
};

const TRUNCATE_LENGTH = 40;

/** Truncates a display string past {@link TRUNCATE_LENGTH}, reporting whether it did — callers use that to decide whether the untruncated value needs a `title` tooltip. */
const truncateForDisplay = (value: string): { display: string; isTruncated: boolean } => {
  if (value.length <= TRUNCATE_LENGTH) {
    return { display: value, isTruncated: false };
  }

  return { display: `${value.slice(0, TRUNCATE_LENGTH)}…`, isTruncated: true };
};

/** string/number/boolean → `String`; object with a string `name` → the name; other object/array → JSON; `undefined` → `null` (caller skips the row). Untruncated — pass through {@link truncateForDisplay} for display. */
const stringifyResolvedValue = (value: unknown): string | null => {
  if (value === undefined) {
    return null;
  }

  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  if (value !== null && typeof value === 'object' && typeof (value as { name?: unknown }).name === 'string') {
    return (value as { name: string }).name;
  }

  return JSON.stringify(value);
};

const getDistinctProvenanceLabels = (node: PreviewGraphNode, source: GraphPreviewSourceState): string[] => {
  if (!source.getProvenance) {
    return [];
  }

  const labels = new Set<string>();

  for (const field of Object.keys(node.inputs)) {
    const provenance = source.getProvenance(node.id, field);

    if (provenance) {
      labels.add(provenance.label);
    }
  }

  return [...labels];
};

const getEdgesInLine = (node: PreviewGraphNode, graph: WorkflowPreviewGraph | null, t: TFunction): string => {
  const incoming = graph?.edges.filter((edge) => edge.targetNodeId === node.id) ?? [];

  if (incoming.length === 0) {
    return t('graphPreview.edgesInNone');
  }

  const sources = [...new Set(incoming.map((edge) => edge.sourceNodeId))].join(', ');

  return t('graphPreview.edgesIn', { count: incoming.length, sources });
};

const NodeInspector = ({
  node,
  source,
  onBack,
  onProvenanceClick,
}: {
  node: PreviewGraphNode;
  source: GraphPreviewSourceState;
  onBack: () => void;
  onProvenanceClick: () => void;
}) => {
  const { t } = useTranslation();
  const provenanceLabels = getDistinctProvenanceLabels(node, source);
  const outgoingEdges = source.graph?.edges.filter((edge) => edge.sourceNodeId === node.id) ?? [];
  const resolvedFields = Object.entries(node.inputs)
    .map(([field, value]) => ({
      field,
      value: source.resolvedInputOverrides?.[node.id]?.[field] ?? stringifyResolvedValue(value),
    }))
    .filter((entry): entry is { field: string; value: string } => entry.value !== null);

  return (
    <Stack gap="3" p="2">
      <Box alignItems="center" display="flex" gap="2">
        <IconButton aria-label={t('graphPreview.back')} size="2xs" variant="ghost" onClick={onBack}>
          <ArrowLeftIcon />
        </IconButton>
        <Stack gap="0" minW="0">
          <Badge fontFamily="mono" size="xs" w="fit-content">
            {node.type}
          </Badge>
          <Text color="fg.muted" fontSize="2xs" truncate>
            {node.id}
          </Text>
        </Stack>
      </Box>

      {provenanceLabels.length > 0 ? (
        <Stack gap="1">
          <Text color="fg.muted" fontSize="2xs" fontWeight="semibold">
            {t('graphPreview.setBy')}
          </Text>
          <Stack gap="0.5">
            {provenanceLabels.map((label) => (
              <Button
                key={label}
                alignSelf="flex-start"
                fontSize="2xs"
                fontWeight="normal"
                h="auto"
                px="0"
                size="2xs"
                variant="plain"
                onClick={onProvenanceClick}
              >
                {label}
              </Button>
            ))}
          </Stack>
        </Stack>
      ) : null}

      <Stack gap="1">
        <Text color="fg.muted" fontSize="2xs" fontWeight="semibold">
          {t('graphPreview.resolvedInputs')}
        </Text>
        <DataList.Root gap="1.5" orientation="horizontal" size="sm">
          {resolvedFields.map(({ field, value }) => {
            const { display, isTruncated } = truncateForDisplay(value);

            return (
              <DataList.Item key={field}>
                <DataList.ItemLabel fontSize="2xs">{field}</DataList.ItemLabel>
                <DataList.ItemValue fontSize="2xs" minW="0" title={isTruncated ? value : undefined}>
                  {display}
                </DataList.ItemValue>
              </DataList.Item>
            );
          })}
        </DataList.Root>
      </Stack>

      <Stack gap="1">
        <Text color="fg.muted" fontSize="2xs" fontWeight="semibold">
          {t('graphPreview.edges')}
        </Text>
        <Stack gap="0.5">
          <Text fontSize="2xs">{getEdgesInLine(node, source.graph, t)}</Text>
          {outgoingEdges.map((edge) => (
            <Text key={edge.id} fontSize="2xs">
              {t('graphPreview.edgesOut', { field: edge.sourceField, target: edge.targetNodeId })}
            </Text>
          ))}
        </Stack>
      </Stack>
    </Stack>
  );
};
