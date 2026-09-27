import type {
  InvocationTemplate,
  ProjectGraphState,
  WorkflowInvocationNode,
  WorkflowNode,
} from '@features/workflow/contracts';
import type { ChangeEvent } from 'react';

import { Box, Flex, HStack, Image, Stack, Text, Textarea } from '@chakra-ui/react';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { workflowSelectionStore } from '@features/workflow/ui/editor/selectionStore';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  useWorkflowHostCommands,
  useWorkflowNodeExecutionState,
  useWorkflowProjectSelector,
} from '@features/workflow/ui/WorkflowUiContext';
import {
  formatOutputFieldValue,
  getNodeUpdateStatus,
  getWorkflowBatchGroupId,
  isWorkflowBatchNodeType,
} from '@features/workflow/utility';
import { Button, JsonPreview, Scrollable, Tabs } from '@platform/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type InspectorTab = 'details' | 'outputs' | 'data' | 'template';

const getInspectorTab = (values: Record<string, unknown>): InspectorTab =>
  values.inspectorTab === 'outputs' || values.inspectorTab === 'data' || values.inspectorTab === 'template'
    ? values.inspectorTab
    : 'details';

const DetailRow = ({ label, value }: { label: string; value: string }) => (
  <HStack align="start" gap="2">
    <Text color="fg.subtle" flexShrink={0} fontSize="2xs" minW="16">
      {label}
    </Text>
    <Text fontSize="2xs" minW="0" wordBreak="break-word">
      {value}
    </Text>
  </HStack>
);

const JsonBlock = ({ label, value }: { label: string; value: unknown }) => <JsonPreview label={label} value={value} />;

/** " (Group 1)" or " (no group)" after a batch node's title; empty for every other node. */
const getBatchGroupSuffix = (node: WorkflowInvocationNode, t: ReturnType<typeof useTranslation>['t']): string => {
  if (!isWorkflowBatchNodeType(node.data.type)) {
    return '';
  }

  const groupId = getWorkflowBatchGroupId(node);

  return ` (${groupId === 'None' ? t('nodes.noBatchGroup') : groupId})`;
};

/** The version row's action: update in place when the template allows it, otherwise say why not. */
const NodeUpdateRow = ({ node, template }: { node: WorkflowInvocationNode; template: InvocationTemplate }) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const status = getNodeUpdateStatus(node, template);
  // The button leaves once the node is current; keyboard focus steps to the inspector's active tab first.
  const onUpdate = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.currentTarget
        .closest('[data-node-inspector]')
        ?.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]')
        ?.focus();
      editGraph({ nodeIds: [node.id], templates: { [template.type]: template }, type: 'updateNodes' });
    },
    [editGraph, node.id, template]
  );

  if (status === 'current') {
    return null;
  }

  const versions = { from: node.data.version, to: template.version };

  return (
    <HStack align="start" gap="2">
      {/* Aligned with the detail values: an empty label cell the width of the label column. */}
      <Box flexShrink={0} minW="16" />
      {status === 'updatable' ? (
        <Button size="2xs" variant="outline" onClick={onUpdate}>
          {t('nodes.updateNodeTo', { version: template.version })}
        </Button>
      ) : (
        <Text color="fg.warning" fontSize="2xs">
          {status === 'newer'
            ? t('nodes.nodeNewerThanBackend', versions)
            : t('nodes.nodeVersionIncompatible', versions)}
        </Text>
      )}
    </HStack>
  );
};

const DetailsTab = ({ node, template }: { node: WorkflowInvocationNode; template: InvocationTemplate | undefined }) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const onNotesChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) =>
      editGraph({ nodeId: node.id, notes: event.currentTarget.value, type: 'setNodeNotes' }),
    [editGraph, node.id]
  );

  return (
    <Stack gap="2">
      <DetailRow
        label={t('widgets.workflow.title')}
        value={`${node.data.label || template?.title || node.data.type}${getBatchGroupSuffix(node, t)}`}
      />
      <DetailRow label={t('widgets.workflow.type')} value={node.data.type} />
      <DetailRow label={t('widgets.workflow.version')} value={node.data.version} />
      {template ? <NodeUpdateRow node={node} template={template} /> : null}
      {template ? <DetailRow label={t('widgets.workflow.class')} value={template.classification} /> : null}
      {template ? <DetailRow label={t('widgets.workflow.pack')} value={template.nodePack} /> : null}
      {template?.description ? <DetailRow label={t('widgets.workflow.about')} value={template.description} /> : null}
      <Stack gap="1">
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.workflow.notes')}
        </Text>
        <Textarea
          aria-label={t('widgets.workflow.nodeNotes')}
          fontSize="2xs"
          minH="3rem"
          placeholder={t('widgets.workflow.nodeNotesPlaceholder')}
          resize="vertical"
          size="xs"
          value={node.data.notes}
          onChange={onNotesChange}
        />
      </Stack>
    </Stack>
  );
};

const OutputsTab = ({ nodeId, template }: { nodeId: string; template: InvocationTemplate | undefined }) => {
  const { t } = useTranslation();
  const execution = useWorkflowNodeExecutionState(nodeId);
  const outputs = template ? Object.values(template.outputs) : [];

  if (!execution) {
    return (
      <Stack gap="2">
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.workflow.noRunRecorded')}
        </Text>
        {outputs.length > 0 ? (
          <Stack gap="1">
            <Text color="fg.subtle" fontSize="2xs">
              {t('widgets.workflow.declaredOutputs')}
            </Text>
            {outputs.map((output) => (
              <DetailRow key={output.name} label={output.title} value={output.type.name} />
            ))}
          </Stack>
        ) : null}
      </Stack>
    );
  }

  // The store keeps the previous run's result while a node runs or fails;
  // showing it under a "Running…" or "Failed" line would pass it off as this run's.
  const hasResult =
    execution.status === 'completed' && execution.latestOutput !== null && execution.latestOutput !== undefined;

  return (
    <Stack gap="2">
      <DetailRow
        label={t('widgets.workflow.status')}
        value={
          execution.status === 'failed' && execution.error
            ? `${t('widgets.workflow.runStatus.failed')} — ${execution.error}`
            : t(`widgets.workflow.runStatus.${execution.status}`)
        }
      />
      {execution.status === 'completed' && execution.outputImageUrl ? (
        <Image
          alt={t('widgets.workflow.outputImage')}
          bg="bg.muted"
          h="10rem"
          objectFit="contain"
          rounded="sm"
          src={execution.outputImageUrl}
          w="full"
        />
      ) : null}
      {hasResult ? (
        <>
          {outputs.map((output) => (
            <DetailRow
              key={output.name}
              label={output.title}
              value={formatOutputFieldValue(execution.latestOutput, output.name)?.full ?? '—'}
            />
          ))}
          <JsonBlock label={t('widgets.workflow.latestOutput')} value={execution.latestOutput} />
        </>
      ) : null}
    </Stack>
  );
};

const InspectorBody = ({ node, tab }: { node: WorkflowNode; tab: InspectorTab }) => {
  const { t } = useTranslation();
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);

  if (node.type === 'notes' || node.type === 'current_image' || node.type === 'connector') {
    const typeLabel =
      node.type === 'notes'
        ? t('widgets.workflow.notesNode')
        : node.type === 'current_image'
          ? t('widgets.workflow.currentImageNode')
          : t('widgets.workflow.connectorNode');

    return tab === 'data' ? (
      <JsonBlock label={t('widgets.workflow.nodeData')} value={node.data} />
    ) : (
      <DetailRow label={t('widgets.workflow.type')} value={typeLabel} />
    );
  }

  const template = templates[node.data.type];

  switch (tab) {
    case 'details':
      return <DetailsTab node={node} template={template} />;
    case 'outputs':
      return <OutputsTab nodeId={node.id} template={template} />;
    case 'data':
      return <JsonBlock label={t('widgets.workflow.nodeData')} value={node.data} />;
    case 'template':
      return template ? (
        <JsonBlock label={t('widgets.workflow.nodeTemplate')} value={template} />
      ) : (
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.workflow.noTemplateKnown', { type: node.data.type })}
        </Text>
      );
  }
};

export const NodeInspector = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const { t } = useTranslation();
  const workflowWidgetValues = useWorkflowProjectSelector((project) => project.workflowValues);
  const { widgets } = useWorkflowHostCommands();
  const selectedNodeIds = workflowSelectionStore.useSelector((snapshot) => snapshot.selectedNodeIds);
  const tab = getInspectorTab(workflowWidgetValues);
  const selectedNode = projectGraph.nodes.find((node) => node.id === selectedNodeIds[0]);
  const onTabValueChange = useCallback(
    (event: { value: string }) => widgets.patchValues('workflow', { inspectorTab: event.value }),
    [widgets]
  );

  return (
    <Flex data-node-inspector="" direction="column" h="full" minH="0">
      <HStack flexShrink={0} justify="space-between" px="2" h={10} borderBottomWidth={1}>
        <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
          {t('widgets.workflow.nodeInspector')}
        </Text>
        <Tabs.Root size="sm" value={tab} variant="outline" mb="-1" onValueChange={onTabValueChange}>
          <Tabs.List>
            {(['details', 'outputs', 'data', 'template'] as const).map((value) => (
              <Tabs.Trigger key={value} fontSize="2xs" textTransform="capitalize" value={value}>
                {t(`widgets.workflow.inspectorTabs.${value}`)}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </Tabs.Root>
      </HStack>
      <Scrollable flex="1" label={t('widgets.workflow.selectedNodeInspector')} minH="0">
        <Stack p="3">
          {selectedNode ? (
            <InspectorBody node={selectedNode} tab={tab} />
          ) : (
            <Text color="fg.subtle" fontSize="2xs">
              {t('widgets.workflow.selectNodeToInspect')}
            </Text>
          )}
        </Stack>
      </Scrollable>
    </Flex>
  );
};
