import type {
  InvocationTemplate,
  ProjectGraphState,
  WorkflowInvocationNode,
  WorkflowNode,
} from '@features/workflow/contracts';
import type { ChangeEvent } from 'react';

import { Flex, HStack, Stack, Text, Textarea } from '@chakra-ui/react';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { workflowSelectionStore } from '@features/workflow/ui/editor/selectionStore';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowHostCommands, useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { JsonPreview, Scrollable, Tabs } from '@platform/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Edit-mode inspector for the editor's selected node, with the legacy tab
 * set: Details (summary + notes), Outputs, Data (raw node data), Template
 * (the node's backend definition). Sizing is owned by the panel's Splitter.
 */

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
      <DetailRow label={t('widgets.workflow.title')} value={node.data.label || template?.title || node.data.type} />
      <DetailRow label={t('widgets.workflow.type')} value={node.data.type} />
      <DetailRow label={t('widgets.workflow.version')} value={node.data.version} />
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

const OutputsTab = ({ template }: { template: InvocationTemplate | undefined }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="2">
      <Text color="fg.subtle" fontSize="2xs">
        {t('widgets.workflow.runOutputsNotRecorded')}
      </Text>
      {template ? (
        <Stack gap="1">
          <Text color="fg.subtle" fontSize="2xs">
            {t('widgets.workflow.declaredOutputs')}
          </Text>
          {Object.values(template.outputs).map((output) => (
            <DetailRow key={output.name} label={output.title} value={output.type.name} />
          ))}
        </Stack>
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
      return <OutputsTab template={template} />;
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
    <Flex direction="column" h="full" minH="0">
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
