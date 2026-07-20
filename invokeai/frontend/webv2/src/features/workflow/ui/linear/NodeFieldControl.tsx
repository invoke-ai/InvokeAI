import { Alert, Field, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { isInvocationNode, type NodeFieldFormElement, type ProjectGraphState } from '@features/workflow/contracts';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { WorkflowFieldInput } from '@features/workflow/ui/fields/WorkflowFieldInput';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import {
  cloneWorkflowFieldDefault,
  getResolvedWorkflowEdges,
  getWorkflowFieldInvalidReason,
  isDirectInputField,
  isWorkflowFieldValueDefault,
} from '@features/workflow/utility';
import { FieldLabel, IconButton, Tooltip } from '@platform/ui';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type ChangeEvent } from 'react';

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
  const { editGraph } = useProjectGraphCommands();
  const { fieldName, instance, invocationNode, nodeContext, nodeId, template } = useNodeFieldBinding(
    element,
    projectGraph
  );
  // While the label input is focused it edits a draft seeded from the
  // *displayed* label, so an unset override starts from the template title
  // instead of an empty box.
  const [draftLabel, setDraftLabel] = useState<string | null>(null);

  const isConnected = getResolvedWorkflowEdges(projectGraph.nodes, projectGraph.edges).some(
    (edge) => edge.target === nodeId && edge.targetHandle === fieldName
  );
  const label = instance?.label || template?.title || '';
  const description = instance?.description || template?.description;
  const invalidReason = template
    ? getWorkflowFieldInvalidReason({ isConnected, template, value: instance?.value })
    : null;
  const isInvalid = invalidReason !== null;
  const canReset =
    !!template &&
    !isConnected &&
    isDirectInputField(template) &&
    !isWorkflowFieldValueDefault(template, instance?.value);
  const labelInputId = `${element.id}-label-input`;
  const valueInputId = `${element.id}-value`;
  const onResetClick = useCallback(
    () =>
      editGraph({
        fieldName,
        nodeId,
        type: 'setFieldValue',
        value: template ? cloneWorkflowFieldDefault(template) : undefined,
      }),
    [editGraph, fieldName, nodeId, template]
  );
  const onLabelBlur = useCallback(() => setDraftLabel(null), []);
  const onLabelChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      setDraftLabel(event.currentTarget.value);
      editGraph({ fieldName, label: event.currentTarget.value, nodeId, type: 'setFieldLabel' });
    },
    [editGraph, fieldName, nodeId]
  );
  const onLabelFocus = useCallback(() => setDraftLabel(label), [label]);
  const onValueChange = useCallback(
    (value: unknown) => editGraph({ fieldName, nodeId, type: 'setFieldValue', value }),
    [editGraph, fieldName, nodeId]
  );
  const resetAriaLabel = useMemo(() => `Reset ${label} to default value`, [label]);

  if (!invocationNode || !template) {
    return (
      <Alert.Root status="error" size="sm" variant="surface">
        <Alert.Indicator />
        <Alert.Title>This field no longer exists in the project graph.</Alert.Title>
      </Alert.Root>
    );
  }
  const resetButton = canReset ? (
    <Tooltip content="Reset to default value">
      <IconButton
        aria-label={resetAriaLabel}
        color="fg.subtle"
        flexShrink={0}
        size="2xs"
        title="Reset to default value"
        variant="ghost"
        onClick={onResetClick}
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
              onBlur={onLabelBlur}
              onChange={onLabelChange}
              onFocus={onLabelFocus}
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
            onChange={onValueChange}
          />
        )}
        {invalidReason ? <Field.ErrorText fontSize="2xs">{invalidReason}</Field.ErrorText> : null}
      </Stack>
    </Field.Root>
  );
};
