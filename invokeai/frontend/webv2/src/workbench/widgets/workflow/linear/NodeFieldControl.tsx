import type { NodeFieldFormElement, ProjectGraphState } from '@workbench/workflows/types';

import { Alert, Field, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { FieldLabel, IconButton, Tooltip } from '@workbench/components/ui';
import { WorkflowFieldInput } from '@workbench/widgets/workflow/fields/WorkflowFieldInput';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { getResolvedWorkflowEdges } from '@workbench/workflows/connectors';
import {
  cloneWorkflowFieldDefault,
  getWorkflowFieldInvalidReason,
  isDirectInputField,
  isWorkflowFieldValueDefault,
} from '@workbench/workflows/fields';
import { useInvocationTemplatesSelector } from '@workbench/workflows/templates';
import { isInvocationNode } from '@workbench/workflows/types';
import { RotateCcwIcon } from 'lucide-react';
import { useState, type ChangeEvent } from 'react';

/**
 * One exposed node field, shared by the Linear UI's view mode and the form
 * builder: label (optionally editable), optional description, and the live
 * input — or a note when the field is driven by a graph connection.
 */
/** Resolves a form element's node, field instance, and input template against the document. */
export const useNodeFieldBinding = (element: NodeFieldFormElement, projectGraph: ProjectGraphState) => {
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
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
      <Alert.Root status="error" size="sm" variant="surface">
        <Alert.Indicator />
        <Alert.Title>This field no longer exists in the project graph.</Alert.Title>
      </Alert.Root>
    );
  }

  const isConnected = getResolvedWorkflowEdges(projectGraph.nodes, projectGraph.edges).some(
    (edge) => edge.target === nodeId && edge.targetHandle === fieldName
  );
  const label = instance?.label || template.title;
  const description = instance?.description || template.description;
  const invalidReason = getWorkflowFieldInvalidReason({ isConnected, template, value: instance?.value });
  const isInvalid = invalidReason !== null;
  const canReset =
    !isConnected && isDirectInputField(template) && !isWorkflowFieldValueDefault(template, instance?.value);
  const labelInputId = `${element.id}-label-input`;
  const valueInputId = `${element.id}-value`;
  const resetButton = canReset ? (
    <Tooltip content="Reset to default value">
      <IconButton
        aria-label={`Reset ${label} to default value`}
        color="fg.subtle"
        flexShrink={0}
        size="2xs"
        title="Reset to default value"
        variant="ghost"
        onClick={() =>
          dispatch({
            action: { fieldName, nodeId, type: 'setFieldValue', value: cloneWorkflowFieldDefault(template) },
            type: 'applyProjectGraphAction',
          })
        }
      >
        <Icon as={RotateCcwIcon} boxSize="3" />
      </IconButton>
    </Tooltip>
  ) : null;

  return (
    <Field.Root invalid={isInvalid} minW="0" w="full">
      <Stack gap="1" minW="0" w="full">
        {isLabelEditable ? (
          <HStack gap="1" minW="0" w="full">
            <Input
              aria-label="Field label"
              color={isInvalid ? 'fg.error' : 'fg.muted'}
              fontSize="2xs"
              fontWeight="600"
              h="5"
              id={labelInputId}
              placeholder={template.title}
              size="2xs"
              textTransform="uppercase"
              value={draftLabel ?? label}
              variant="flushed"
              w="full"
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
            {resetButton}
          </HStack>
        ) : (
          <HStack gap="1" minW="0" w="full">
            <Tooltip content={`${nodeContext} → ${template.title}`}>
              <Stack color={isInvalid ? 'fg.error' : undefined} flex="1" gap="0" minW="0">
                <FieldLabel>{label}</FieldLabel>
              </Stack>
            </Tooltip>
            {resetButton}
          </HStack>
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
            id={valueInputId}
            invalid={isInvalid}
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
        {invalidReason ? <Field.ErrorText fontSize="2xs">{invalidReason}</Field.ErrorText> : null}
      </Stack>
    </Field.Root>
  );
};
