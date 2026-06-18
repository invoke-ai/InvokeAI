import type {
  InvocationTemplate,
  ProjectGraphState,
  WorkflowInvocationNode,
  WorkflowNode,
} from '@workbench/workflows/types';
import type { ChangeEvent } from 'react';

import { Flex, HStack, Stack, Text, Textarea } from '@chakra-ui/react';
import { JsonPreview, Scrollable, Tabs } from '@workbench/components/ui';
import { workflowSelectionStore } from '@workbench/widgets/workflow/editor/selectionStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';

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
  const dispatch = useWorkbenchDispatch();

  return (
    <Stack gap="2">
      <DetailRow label="Title" value={node.data.label || template?.title || node.data.type} />
      <DetailRow label="Type" value={node.data.type} />
      <DetailRow label="Version" value={node.data.version} />
      {template ? <DetailRow label="Class" value={template.classification} /> : null}
      {template ? <DetailRow label="Pack" value={template.nodePack} /> : null}
      {template?.description ? <DetailRow label="About" value={template.description} /> : null}
      <Stack gap="1">
        <Text color="fg.subtle" fontSize="2xs">
          Notes
        </Text>
        <Textarea
          aria-label="Node notes"
          fontSize="2xs"
          minH="3rem"
          placeholder="Notes about this node…"
          resize="vertical"
          size="xs"
          value={node.data.notes}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
            dispatch({
              action: { nodeId: node.id, notes: event.currentTarget.value, type: 'setNodeNotes' },
              type: 'applyProjectGraphAction',
            })
          }
        />
      </Stack>
    </Stack>
  );
};

const OutputsTab = ({ template }: { template: InvocationTemplate | undefined }) => (
  <Stack gap="2">
    <Text color="fg.subtle" fontSize="2xs">
      Run outputs are not recorded per node yet — results currently route to the Gallery or Canvas. This tab will show
      the node's outputs from its most recent run once run records land.
    </Text>
    {template ? (
      <Stack gap="1">
        <Text color="fg.subtle" fontSize="2xs">
          Declared outputs
        </Text>
        {Object.values(template.outputs).map((output) => (
          <DetailRow key={output.name} label={output.title} value={output.type.name} />
        ))}
      </Stack>
    ) : null}
  </Stack>
);

const InspectorBody = ({ node, tab }: { node: WorkflowNode; tab: InspectorTab }) => {
  const { templates } = useInvocationTemplatesSnapshot();

  if (node.type === 'notes' || node.type === 'current_image' || node.type === 'connector') {
    const typeLabel =
      node.type === 'notes' ? 'Notes node' : node.type === 'current_image' ? 'Current Image node' : 'Connector node';

    return tab === 'data' ? (
      <JsonBlock label="Node data" value={node.data} />
    ) : (
      <DetailRow label="Type" value={typeLabel} />
    );
  }

  const template = templates[node.data.type];

  switch (tab) {
    case 'details':
      return <DetailsTab node={node} template={template} />;
    case 'outputs':
      return <OutputsTab template={template} />;
    case 'data':
      return <JsonBlock label="Node data" value={node.data} />;
    case 'template':
      return template ? (
        <JsonBlock label="Node template" value={template} />
      ) : (
        <Text color="fg.subtle" fontSize="2xs">
          No template is known for "{node.data.type}" on this backend.
        </Text>
      );
  }
};

export const NodeInspector = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const workflowWidgetValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'workflow'));
  const dispatch = useWorkbenchDispatch();
  const { selectedNodeIds } = workflowSelectionStore.useSnapshot();
  const tab = getInspectorTab(workflowWidgetValues);
  const selectedNode = projectGraph.nodes.find((node) => node.id === selectedNodeIds[0]);

  return (
    <Flex direction="column" h="full" minH="0">
      <HStack flexShrink={0} justify="space-between" px="2" h={10} borderBottomWidth={1}>
        <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
          Node Inspector
        </Text>
        <Tabs.Root
          size="sm"
          value={tab}
          variant="outline"
          mb="-1"
          onValueChange={(event) =>
            dispatch({ type: 'patchWidgetValues', values: { inspectorTab: event.value }, widgetId: 'workflow' })
          }
        >
          <Tabs.List>
            {(['details', 'outputs', 'data', 'template'] as const).map((value) => (
              <Tabs.Trigger key={value} fontSize="2xs" textTransform="capitalize" value={value}>
                {value}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        </Tabs.Root>
      </HStack>
      <Scrollable flex="1" label="Selected node inspector" minH="0">
        <Stack p="3">
          {selectedNode ? (
            <InspectorBody node={selectedNode} tab={tab} />
          ) : (
            <Text color="fg.subtle" fontSize="2xs">
              Select a node in the Workflow editor to inspect it.
            </Text>
          )}
        </Stack>
      </Scrollable>
    </Flex>
  );
};
