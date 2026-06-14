import { HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { TriangleAlertIcon } from 'lucide-react';
import { useState, type ChangeEvent } from 'react';

import { FieldLabel } from '../../../components/ui/Field';
import { Tooltip } from '../../../components/ui/Tooltip';
import { useWorkbenchDispatch } from '../../../WorkbenchContext';
import { useInvocationTemplatesSnapshot } from '../../../workflows/templates';
import type { NodeFieldFormElement, ProjectGraphState } from '../../../workflows/types';
import { isInvocationNode } from '../../../workflows/types';
import { WorkflowFieldInput } from '../fields/WorkflowFieldInput';

/**
 * One exposed node field, shared by the Linear UI's view mode and the form
 * builder: label (optionally editable), optional description, and the live
 * input — or a note when the field is driven by a graph connection.
 */
/** Resolves a form element's node, field instance, and input template against the document. */
export const useNodeFieldBinding = (element: NodeFieldFormElement, projectGraph: ProjectGraphState) => {
  const { templates } = useInvocationTemplatesSnapshot();
  const { fieldName, nodeId } = element.data.fieldIdentifier;
  const node = projectGraph.nodes.find((candidate) => candidate.id === nodeId);
  const invocationNode = node && isInvocationNode(node) ? node : null;
  const template = invocationNode ? templates[invocationNode.data.type]?.inputs[fieldName] : undefined;
  const instance = invocationNode?.data.inputs[fieldName];
  const nodeContext = invocationNode
    ? invocationNode.data.label || templates[invocationNode.data.type]?.title || invocationNode.data.type
    : '';

  return { fieldName, instance, invocationNode, nodeContext, nodeId, template };
};

export const NodeFieldControl = ({
  element,
  isLabelEditable = false,
  projectGraph,
}: {
  element: NodeFieldFormElement;
  /** Builder mode renders the label as an input bound to the field instance label. */
  isLabelEditable?: boolean;
  projectGraph: ProjectGraphState;
}) => {
  const dispatch = useWorkbenchDispatch();
  const { fieldName, instance, invocationNode, nodeContext, nodeId, template } = useNodeFieldBinding(
    element,
    projectGraph
  );
  // While the label input is focused it edits a draft seeded from the
  // *displayed* label, so an unset override starts from the template title
  // instead of an empty box.
  const [draftLabel, setDraftLabel] = useState<string | null>(null);

  if (!invocationNode || !template) {
    return (
      <HStack gap="1.5">
        <Icon as={TriangleAlertIcon} boxSize="3" color="fg.error" />
        <Text color="fg.subtle" fontSize="2xs">
          This field no longer exists in the project graph.
        </Text>
      </HStack>
    );
  }

  const isConnected = projectGraph.edges.some((edge) => edge.target === nodeId && edge.targetHandle === fieldName);
  const label = instance?.label || template.title;
  const description = instance?.description || template.description;

  return (
    <Stack gap="1">
      {isLabelEditable ? (
        <Input
          aria-label="Field label"
          color="fg.muted"
          fontSize="2xs"
          fontWeight="600"
          h="5"
          placeholder={template.title}
          size="2xs"
          textTransform="uppercase"
          value={draftLabel ?? label}
          variant="flushed"
          onBlur={() => setDraftLabel(null)}
          onChange={(event: ChangeEvent<HTMLInputElement>) => {
            setDraftLabel(event.currentTarget.value);
            dispatch({
              action: { fieldName, label: event.currentTarget.value, nodeId, type: 'setFieldLabel' },
              type: 'applyProjectGraphAction',
            });
          }}
          onFocus={() => setDraftLabel(label)}
        />
      ) : (
        <Tooltip content={`${nodeContext} → ${template.title}`}>
          <Stack gap="0" w="fit-content">
            <FieldLabel>{label}</FieldLabel>
          </Stack>
        </Tooltip>
      )}
      {element.data.showDescription && description ? (
        <Text color="fg.subtle" fontSize="2xs">
          {description}
        </Text>
      ) : null}
      {isConnected ? (
        <Text color="fg.subtle" fontSize="2xs">
          Driven by a graph connection.
        </Text>
      ) : (
        <WorkflowFieldInput
          template={template}
          value={instance?.value}
          onChange={(value) =>
            dispatch({
              action: { fieldName, nodeId, type: 'setFieldValue', value },
              type: 'applyProjectGraphAction',
            })
          }
        />
      )}
    </Stack>
  );
};
