import type { SeedMode } from '@platform/core/seed';

import { Alert, Box, Field, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { isInvocationNode, type NodeFieldFormElement, type ProjectGraphState } from '@features/workflow/contracts';
import { getWorkflowFieldSeedMode, isSeedInputField } from '@features/workflow/graph';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { WorkflowFieldInput } from '@features/workflow/ui/fields/WorkflowFieldInput';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import {
  applyNodeFieldFormSettings,
  cloneWorkflowFieldDefault,
  getEffectiveWorkflowFieldDescription,
  getNodeFieldFormSettings,
  getRandomWorkflowFieldValue,
  getResolvedWorkflowEdges,
  getWorkflowFieldInvalidReason,
  isDirectInputField,
  isShuffleableField,
  isWorkflowFieldValueDefault,
  getWorkflowBatchGroupId,
  isWorkflowBatchNodeType,
} from '@features/workflow/utility';
import { FieldLabel, IconButton, Tooltip } from '@platform/ui';
import { DicesIcon, RotateCcwIcon } from 'lucide-react';
import { useCallback, useMemo, useState, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { LinkifiedText } from './LinkifiedText';

/** Resolves a form element's node, field instance, and input template against the document. */
export const useNodeFieldBinding = (element: NodeFieldFormElement, projectGraph: ProjectGraphState) => {
  const { t } = useTranslation();
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const { fieldName, nodeId } = element.data.fieldIdentifier;
  const node = projectGraph.nodes.find((candidate) => candidate.id === nodeId);
  const invocationNode = node && isInvocationNode(node) ? node : null;
  const template = invocationNode
    ? (invocationNode.data.dynamicInputTemplates?.[fieldName] ?? templates[invocationNode.data.type]?.inputs[fieldName])
    : undefined;
  const instance = invocationNode?.data.inputs[fieldName];
  const formSettings = template ? getNodeFieldFormSettings(element.data.settings, template) : null;
  // The authored bounds narrow the template once here; validation, shuffle and the control all read the result.
  const effectiveTemplate = template ? applyNodeFieldFormSettings(template, formSettings) : undefined;
  const groupId =
    invocationNode && isWorkflowBatchNodeType(invocationNode.data.type)
      ? getWorkflowBatchGroupId(invocationNode)
      : null;
  const groupSuffix = groupId === null ? '' : ` (${groupId === 'None' ? t('nodes.noBatchGroup') : groupId})`;
  const nodeContext = invocationNode
    ? `${invocationNode.data.label || templates[invocationNode.data.type]?.title || invocationNode.data.type}${groupSuffix}`
    : '';

  return { effectiveTemplate, fieldName, formSettings, instance, invocationNode, nodeContext, nodeId, template };
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
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const { effectiveTemplate, fieldName, formSettings, instance, invocationNode, nodeContext, nodeId, template } =
    useNodeFieldBinding(element, projectGraph);
  // The form stays mounted across a workflow switch; a value that arrives later (an upload) must still land in
  // the workflow this control rendered.
  const target = useMemo(() => ({ projectId, workflowId: projectGraph.id }), [projectId, projectGraph.id]);
  // Seed focused label drafts from displayed text, including template fallback.
  const [draftLabel, setDraftLabel] = useState<string | null>(null);

  const isConnected = getResolvedWorkflowEdges(projectGraph.nodes, projectGraph.edges).some(
    (edge) => edge.target === nodeId && edge.targetHandle === fieldName
  );
  const label = instance?.label || template?.title || '';
  const description = getEffectiveWorkflowFieldDescription(instance, template);
  const invalidReason = effectiveTemplate
    ? getWorkflowFieldInvalidReason({ isConnected, template: effectiveTemplate, value: instance?.value })
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
    (value: unknown) => editGraph({ fieldName, nodeId, type: 'setFieldValue', value }, target),
    [editGraph, fieldName, nodeId, target]
  );
  const onSeedModeChange = useCallback(
    (seedMode: SeedMode) => editGraph({ fieldName, nodeId, seedMode, type: 'setFieldSeedMode' }),
    [editGraph, fieldName, nodeId]
  );
  // Read the narrowed template: authored bounds and step can leave no value to shuffle to.
  const showsShuffle =
    element.data.showShuffle &&
    !!template &&
    !!effectiveTemplate &&
    !isConnected &&
    isShuffleableField(effectiveTemplate) &&
    !isSeedInputField(template);
  const onShuffleClick = useCallback(() => {
    if (effectiveTemplate) {
      editGraph({ fieldName, nodeId, type: 'setFieldValue', value: getRandomWorkflowFieldValue(effectiveTemplate) });
    }
  }, [editGraph, effectiveTemplate, fieldName, nodeId]);

  if (!invocationNode || !template || !effectiveTemplate) {
    return (
      <Alert.Root status="error" variant="surface">
        <Alert.Indicator />
        <Alert.Title>{t('nodes.fieldUnavailable')}</Alert.Title>
      </Alert.Root>
    );
  }
  const resetButton = canReset ? (
    <Tooltip content={t('nodes.resetToDefault')}>
      <IconButton
        aria-label={t('nodes.resetFieldToDefault', { label })}
        color="fg.subtle"
        flexShrink={0}
        size="sm"
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
              aria-label={t('nodes.fieldLabel')}
              color={isInvalid ? 'fg.error' : 'fg.muted'}
              fontSize="xs"
              fontWeight="600"
              h="5"
              id={labelInputId}
              placeholder={template.title}
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
          <LinkifiedText color="fg.subtle" fontSize="xs" whiteSpace="pre-wrap">
            {description}
          </LinkifiedText>
        ) : null}
        {isConnected ? (
          <Text color="fg.subtle" fontSize="xs">
            {t('nodes.providedByConnection')}
          </Text>
        ) : (
          <HStack alignItems="start" gap="1" minW="0" w="full">
            <Box flex="1" minW="0">
              <WorkflowFieldInput
                formSettings={formSettings}
                id={valueInputId}
                invalid={isInvalid}
                nodeId={nodeId}
                seedMode={getWorkflowFieldSeedMode(instance)}
                template={effectiveTemplate}
                value={instance?.value}
                workflowId={projectGraph.id}
                onChange={onValueChange}
                onSeedModeChange={onSeedModeChange}
              />
            </Box>
            {showsShuffle ? (
              <Tooltip content={t('common.shuffle')}>
                <IconButton
                  aria-label={t('nodes.shuffleField', { label })}
                  color="fg.muted"
                  flexShrink={0}
                  variant="outline"
                  onClick={onShuffleClick}
                >
                  <Icon as={DicesIcon} boxSize="3.5" />
                </IconButton>
              </Tooltip>
            ) : null}
          </HStack>
        )}
        {invalidReason ? <Field.ErrorText fontSize="xs">{invalidReason}</Field.ErrorText> : null}
      </Stack>
    </Field.Root>
  );
};
