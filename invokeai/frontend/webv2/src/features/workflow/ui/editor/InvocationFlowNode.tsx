/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { FieldInputTemplate, FieldOutputTemplate, WorkflowInvocationNode } from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState as NodeExecutionState } from '@features/workflow/ui/contracts';

import {
  Box,
  chakra,
  Checkbox,
  Field,
  Flex,
  HStack,
  Icon,
  IconButton,
  Image,
  Input,
  Stack,
  Text,
} from '@chakra-ui/react';
import { getWorkflowFieldSeedMode, isSeedInputField } from '@features/workflow/graph';
import { FieldDescriptionPopover } from '@features/workflow/ui/fields/FieldDescriptionPopover';
import { WorkflowFieldInput } from '@features/workflow/ui/fields/WorkflowFieldInput';
import {
  getWorkflowNodeBodyProps,
  getWorkflowNodeHandleStyle,
  getWorkflowNodeHeaderProps,
  getWorkflowNodeShellProps,
  WORKFLOW_NODE_DENSITY,
  WORKFLOW_NODE_SURFACE_TOKEN,
  WorkflowNodeInfoIcon,
  WorkflowNodeOutcomeIcon,
  type WorkflowNodeOutcome,
} from '@features/workflow/ui/nodeChrome';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNodeExecutionState } from '@features/workflow/ui/WorkflowUiContext';
import { setNodePreviewCollapsed, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import {
  CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX,
  cloneWorkflowFieldDefault,
  formatOutputFieldValue,
  getEffectiveWorkflowFieldDescription,
  getFieldTypeLabel,
  getNodeUpdateStatus,
  getOutputFieldNamesByScope,
  getWorkflowBatchGroupId,
  isWorkflowBatchNodeType,
  getOutputFieldRows,
  getWorkflowFieldInvalidReason,
  isDirectInputField,
  isExposableField,
  isWorkflowFieldValueDefault,
} from '@features/workflow/utility';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Tooltip } from '@platform/ui';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';
import { ChevronDownIcon, ChevronRightIcon, PinIcon, PinOffIcon, RotateCcwIcon, TriangleAlertIcon } from 'lucide-react';
import { memo, useId, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

import type { InvocationFlowNode as InvocationFlowNodeType, InvocationNodeTemplateView } from './flowAdapters';

import { getHandleTypeTooltip } from './handleTooltip';

const NODE_WIDTH = '18rem';
/** Row padding-x in px; inline handles sit in the label rows, so they pull back out past it. */
const ROW_PADDING_X = 12;
/** Below this viewport zoom, field content renders as skeleton bars (ComfyUI-style) for performance/readability. */
const CONTENT_VISIBILITY_ZOOM = 0.4;

/** True while the viewport is zoomed out far enough that field content is unreadable noise. */
const useIsZoomedOut = (): boolean => useStore((state) => state.transform[2] < CONTENT_VISIBILITY_ZOOM);

/** The node-level loading hint is only useful while a selected child signature is being fetched. */
export const shouldShowCallSavedWorkflowLoadingHint = (node: WorkflowInvocationNode): boolean =>
  node.data.type === 'call_saved_workflow' && node.data.callSavedWorkflowStatus === 'loading';

/** A callable child can be valid without exposing any fields; explain the empty body instead of showing a blank node. */
export const shouldShowCallSavedWorkflowNoExposedFieldsHint = (node: WorkflowInvocationNode): boolean => {
  if (node.data.type !== 'call_saved_workflow' || node.data.callSavedWorkflowStatus !== 'ready') {
    return false;
  }

  const workflowId = node.data.inputs.workflow_id?.value;
  const hasSelectedWorkflow = typeof workflowId === 'string' && workflowId.trim() !== '';
  const hasDynamicFields =
    Object.keys(node.data.dynamicInputTemplates ?? {}).length > 0 ||
    Object.keys(node.data.inputs).some((name) => name.startsWith(CALL_SAVED_WORKFLOW_DYNAMIC_FIELD_PREFIX));

  return hasSelectedWorkflow && !hasDynamicFields;
};

/** Static placeholder bar standing in for text/controls at far zoom. No animation — there may be hundreds. */
const SkeletonBar = ({ h = '2', w }: { h?: string; w?: string }) => <Box bg="bg.emphasized" h={h} rounded="sm" w={w} />;

const hasMissingRequiredInputs = (
  node: WorkflowInvocationNode,
  templateInputs: FieldInputTemplate[],
  connectedFieldNames: Set<string>
): boolean =>
  templateInputs.some(
    (inputTemplate) =>
      getWorkflowFieldInvalidReason({
        isConnected: connectedFieldNames.has(inputTemplate.name),
        template: inputTemplate,
        value: node.data.inputs[inputTemplate.name]?.value,
      }) !== null
  );

const getExecutionOutcome = (execution: NodeExecutionState | null): WorkflowNodeOutcome | null =>
  execution?.status === 'completed' || execution?.status === 'failed' ? execution.status : null;

export const getNodeExecutionError = (
  node: WorkflowInvocationNode,
  error: string | null | undefined,
  translate: (key: string, options: { error: string }) => string
): string | null | undefined => {
  if (!error || node.data.type !== 'call_saved_workflow') {
    return error;
  }

  return translate('nodes.childWorkflowError', { error });
};

const NodeShell = ({
  hasMissingRequiredInput,
  children,
  isMissing,
  isOutdated,
  isRunning,
  outcome,
  selected,
}: {
  hasMissingRequiredInput?: boolean;
  children: React.ReactNode;
  isMissing?: boolean;
  isOutdated?: boolean;
  isRunning?: boolean;
  outcome?: WorkflowNodeOutcome | null;
  selected: boolean;
}) => {
  const isInvalid = isMissing || hasMissingRequiredInput;

  return (
    <Box
      w={NODE_WIDTH}
      {...getWorkflowNodeShellProps({
        invalid: isInvalid,
        outcome,
        outdated: isOutdated,
        running: isRunning,
        selected,
      })}
    >
      {children}
    </Box>
  );
};

/** The header's completed/failed mark, named for screen readers and tooltipped with the failure. */
const NodeOutcomeIcon = ({
  execution,
  node,
}: {
  execution: NodeExecutionState | null;
  node: WorkflowInvocationNode;
}) => {
  const { t } = useTranslation();
  const outcome = getExecutionOutcome(execution);

  if (!outcome) {
    return null;
  }

  return (
    <WorkflowNodeOutcomeIcon
      error={getNodeExecutionError(node, execution?.error, t)}
      label={outcome === 'completed' ? t('nodes.executionCompleted') : t('nodes.executionFailed')}
      outcome={outcome}
    />
  );
};

/** Thin progress strip under the header while the node's invocation executes. */
const NodeProgressStrip = ({ execution }: { execution: NodeExecutionState | null }) => {
  if (execution?.status !== 'running') {
    return null;
  }

  return (
    <Box bg="bg.muted" h="2px" overflow="hidden" position="relative" w="full">
      <Box
        bg="brand.solid"
        h="full"
        transition="width var(--wb-motion-duration-slow) ease"
        w={execution.progress === null ? 'full' : `${Math.round(execution.progress * 100)}%`}
        {...(execution.progress === null
          ? {
              animationDuration: 'var(--wb-motion-duration-slow)',
              animationIterationCount: 'var(--wb-motion-animation-iteration-count)',
              animationName: 'pulse',
            }
          : {})}
      />
    </Box>
  );
};

const NodeTitle = ({ node, title }: { node: WorkflowInvocationNode; title: string }) => {
  const { editGraph } = useProjectGraphCommands();
  const [draftLabel, setDraftLabel] = useState<string | null>(null);

  if (draftLabel !== null) {
    return (
      <Input
        autoFocus
        aria-label="Node label"
        className="nodrag"
        size="2xs"
        value={draftLabel}
        onBlur={() => {
          editGraph({ label: draftLabel.trim(), nodeId: node.id, type: 'setNodeLabel' });
          setDraftLabel(null);
        }}
        onChange={(event: ChangeEvent<HTMLInputElement>) => setDraftLabel(event.currentTarget.value)}
        onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
          if (event.key === 'Enter' || event.key === 'Escape') {
            event.currentTarget.blur();
          }
        }}
      />
    );
  }

  return (
    <MiddleTruncate
      fontWeight="700"
      minW="0"
      text={title}
      title="Double-click to rename"
      // Start editing from the displayed title, including the template fallback for unset labels.
      onDoubleClick={() => setDraftLabel(title)}
    />
  );
};

const getInputModeLabel = (input: FieldInputTemplate['input']): string => {
  if (input === 'connection') {
    return 'Connection only';
  }

  if (input === 'direct') {
    return 'Direct value only';
  }

  return 'Direct value or connection';
};

const InputFieldTooltip = ({
  description,
  isConnected,
  isExposed,
  label,
  template,
}: {
  description: string;
  isConnected: boolean;
  isExposed: boolean;
  label: string;
  template: FieldInputTemplate;
}) => (
  <Stack gap="0.5" maxW="18rem">
    <Text fontWeight="700">{label}</Text>
    <Text color="fg.subtle">Field: {template.name}</Text>
    <Text color="fg.subtle">Type: {getFieldTypeLabel(template.type)}</Text>
    <Text color="fg.subtle">
      {template.required ? 'Required' : 'Optional'} · {getInputModeLabel(template.input)}
    </Text>
    {isConnected ? <Text color="fg.subtle">Connected by graph edge.</Text> : null}
    {isExposed ? <Text color="fg.subtle">Pinned to Linear UI.</Text> : null}
    {description ? <Text>{description}</Text> : null}
  </Stack>
);

const OutputFieldTooltip = ({ template }: { template: FieldOutputTemplate }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="0.5" maxW="18rem">
      <Text fontWeight="700">{template.title}</Text>
      <Text color="fg.subtle">{t('nodes.fieldName', { name: template.name })}</Text>
      <Text color="fg.subtle">{t('nodes.fieldType', { type: getFieldTypeLabel(template.type) })}</Text>
      <Text color="fg.subtle">{t('nodes.output')}</Text>
      {template.description ? <Text>{template.description}</Text> : null}
    </Stack>
  );
};

/** One line naming the node's version against its template's, for the tooltip and the inspector. */
const getUpdateStatusText = (
  t: ReturnType<typeof useTranslation>['t'],
  node: WorkflowInvocationNode,
  template: InvocationNodeTemplateView['template']
): string | null => {
  const status = getNodeUpdateStatus(node, template);
  const versions = { from: node.data.version, to: template.version };

  return status === 'current'
    ? null
    : status === 'updatable'
      ? t('nodes.nodeUpdateAvailable', versions)
      : status === 'newer'
        ? t('nodes.nodeNewerThanBackend', versions)
        : t('nodes.nodeVersionIncompatible', versions);
};

const UPDATE_TOOLTIP_POSITIONING = { placement: 'top-end' } as const;

/** Legacy's group tints, so a zipped group reads as one colour across the canvas. */
const BATCH_GROUP_COLORS: Record<string, string> = {
  'Group 1': 'green.fg',
  'Group 2': 'blue.fg',
  'Group 3': 'purple.fg',
  'Group 4': 'red.fg',
  'Group 5': 'yellow.fg',
};

/** A batch node's group beside its title: zipped groups share a colour, an ungrouped node says so. */
const BatchGroupSuffix = ({ node }: { node: WorkflowInvocationNode }) => {
  const { t } = useTranslation();

  if (!isWorkflowBatchNodeType(node.data.type)) {
    return null;
  }

  const groupId = getWorkflowBatchGroupId(node);

  return (
    <Text color={BATCH_GROUP_COLORS[groupId] ?? 'fg.subtle'} flexShrink={0} fontSize="2xs" fontWeight="600">
      ({groupId === 'None' ? t('nodes.noBatchGroup') : groupId})
    </Text>
  );
};

/** Header mark for a node whose version is not its template's; the tooltip says what can be done about it. */
const NodeUpdateIcon = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => {
  const { t } = useTranslation();
  const label = getUpdateStatusText(t, node, template);

  if (label === null) {
    return null;
  }

  return (
    <Tooltip content={label} positioning={UPDATE_TOOLTIP_POSITIONING} showArrow>
      <Icon aria-label={label} as={TriangleAlertIcon} boxSize="3.5" color="fg.warning" flexShrink={0} role="img" />
    </Tooltip>
  );
};

const NodeInfoTooltipContent = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => {
  const { t } = useTranslation();
  const title = node.data.label ? `${node.data.label} (${template.title})` : template.title;
  const nodePack = node.data.nodePack || template.nodePack;
  const updateStatusText = getUpdateStatusText(t, node, template);

  return (
    <Stack gap="1" maxW="20rem">
      <Text fontWeight="700">{title}</Text>
      <Text color="fg.subtle">{t('nodes.nodeType', { type: template.type })}</Text>
      <Text color="fg.subtle">{t('nodes.nodePackLabel', { name: nodePack })}</Text>
      <Text color="fg.subtle">{t('nodes.nodeVersion', { version: node.data.version })}</Text>
      {updateStatusText ? <Text color="fg.warning">{updateStatusText}</Text> : null}
      <Text color="fg.subtle">{t('nodes.nodeClassification', { classification: template.classification })}</Text>
      <Text color="fg.subtle">{t('nodes.nodeCategory', { category: template.category })}</Text>
      {template.description ? <Text fontStyle="italic">{template.description}</Text> : null}
      {node.data.notes ? <Text>{node.data.notes}</Text> : null}
    </Stack>
  );
};

const NodeInfoIcon = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => (
  <WorkflowNodeInfoIcon
    content={<NodeInfoTooltipContent node={node} template={template} />}
    label={`Show details for ${node.data.label || template.title}`}
  />
);

/** Fixed so a run's differently shaped outputs do not resize the node and shove its neighbours around. */
const NODE_OUTPUT_PREVIEW_HEIGHT = '10rem';

/**
 * The node's latest output image behind a disclosure. The fold is session-lived
 * (`workflowUiStore`), matching the preview itself, so it is not a graph edit.
 */
const NodeOutputPreview = ({
  imageUrl,
  isSkeleton,
  nodeId,
  roundedBottom,
}: {
  imageUrl: string;
  isSkeleton: boolean;
  nodeId: string;
  roundedBottom: boolean;
}) => {
  const { t } = useTranslation();
  const contentId = useId();
  const collapsed = useExternalStoreSelector(workflowUiStore.subscribe, workflowUiStore.getSnapshot, (snapshot) =>
    snapshot.collapsedPreviewNodeIds.has(nodeId)
  );

  return (
    <Box borderBottomRadius={roundedBottom ? 'lg' : 'none'} borderColor="border.subtle" borderTopWidth="1px">
      <chakra.button
        alignItems="center"
        aria-controls={contentId}
        aria-expanded={!collapsed}
        className="nodrag"
        display="flex"
        focusVisibleRing="inside"
        gap="1"
        px={WORKFLOW_NODE_DENSITY.rowPaddingX}
        py="1"
        textAlign="start"
        type="button"
        w="full"
        onClick={() => setNodePreviewCollapsed(nodeId, !collapsed)}
      >
        <Icon as={collapsed ? ChevronRightIcon : ChevronDownIcon} boxSize="3" color="fg.subtle" />
        <Text
          color="fg.subtle"
          fontSize="2xs"
          fontWeight="600"
          letterSpacing="wide"
          lineHeight="1"
          textTransform="uppercase"
        >
          {t('nodes.latestOutput')}
        </Text>
      </chakra.button>
      {/* The frame behind the image keeps a letterboxed output reading as a picture, not a gap. */}
      <Box id={contentId} hidden={collapsed} pb="1.5" px={WORKFLOW_NODE_DENSITY.rowPaddingX}>
        {collapsed ? null : isSkeleton ? (
          <SkeletonBar h={NODE_OUTPUT_PREVIEW_HEIGHT} w="full" />
        ) : (
          <Image
            alt={t('nodes.latestOutputImage')}
            bg={WORKFLOW_NODE_SURFACE_TOKEN}
            draggable={false}
            h={NODE_OUTPUT_PREVIEW_HEIGHT}
            objectFit="contain"
            rounded="sm"
            src={imageUrl}
            w="full"
          />
        )}
      </Box>
    </Box>
  );
};

const NodeFooter = ({ canUseCache, node }: { canUseCache: boolean; node: WorkflowInvocationNode }) => {
  const { editGraph } = useProjectGraphCommands();

  return (
    <Flex
      align="center"
      bg="bg.subtle"
      borderBottomRadius="lg"
      borderColor="border.subtle"
      borderTopWidth="1px"
      className="nodrag"
      gap="3"
      justify="space-between"
      minH="8"
      px="2.5"
      py={WORKFLOW_NODE_DENSITY.rowPaddingY}
    >
      <HStack gap="4">
        {canUseCache ? (
          <Checkbox.Root
            checked={node.data.useCache}
            colorPalette="accent"
            size="xs"
            onCheckedChange={(event) =>
              editGraph({ nodeId: node.id, type: 'setNodeUseCache', useCache: event.checked === true })
            }
          >
            <Checkbox.HiddenInput />
            <Checkbox.Control />
            <Checkbox.Label fontSize="2xs">Use Cache</Checkbox.Label>
          </Checkbox.Root>
        ) : null}
        <Checkbox.Root
          checked={!node.data.isIntermediate}
          colorPalette="accent"
          size="xs"
          onCheckedChange={(event) =>
            editGraph({ isIntermediate: event.checked !== true, nodeId: node.id, type: 'setNodeIsIntermediate' })
          }
        >
          <Checkbox.HiddenInput />
          <Checkbox.Control />
          <Checkbox.Label fontSize="2xs">Save to Gallery</Checkbox.Label>
        </Checkbox.Root>
      </HStack>
    </Flex>
  );
};

/** The upstream node owns the seed now; the local mode waits for a disconnect. A leaf so only this row subscribes to i18n. */
const ProvidedByConnectionNote = () => {
  const { t } = useTranslation();

  return (
    <Text color="fg.subtle" fontSize="2xs" mt="0.5">
      {t('nodes.providedByConnection')}
    </Text>
  );
};

const InputFieldRow = ({
  isConnected,
  isExposed,
  isSkeleton,
  node,
  template,
}: {
  isConnected: boolean;
  isExposed: boolean;
  isSkeleton: boolean;
  node: WorkflowInvocationNode;
  template: FieldInputTemplate;
}) => {
  const { editGraph } = useProjectGraphCommands();
  const instance = node.data.inputs[template.name];
  const fieldIdentifier = { fieldName: template.name, nodeId: node.id };
  const showsControl = !isConnected && isDirectInputField(template);
  const label = instance?.label || template.title;
  const invalidReason = getWorkflowFieldInvalidReason({
    isConnected,
    template,
    value: instance?.value,
  });
  const isInvalid = invalidReason !== null;
  const handleTooltip = getHandleTypeTooltip(template.type);
  const canReset = showsControl && !isWorkflowFieldValueDefault(template, instance?.value);

  if (isSkeleton) {
    return (
      <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
        <HStack gap="1.5" h="5" position="relative">
          {template.input !== 'direct' ? (
            <Tooltip content={handleTooltip} showArrow>
              <Handle
                id={template.name}
                position={Position.Left}
                style={{ ...getWorkflowNodeHandleStyle(template.type, 'left'), left: -ROW_PADDING_X, top: '50%' }}
                type="target"
              />
            </Tooltip>
          ) : null}
          <SkeletonBar w="55%" />
        </HStack>
        {showsControl ? <SkeletonBar h="6" w="full" /> : null}
      </Box>
    );
  }

  return (
    <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY} w="full">
      <Field.Root gap="0" invalid={isInvalid} minW="0" w="full">
        {/* The handle lives inside the label row so it stays centered on the
            label even when the value control below grows the row. */}
        <HStack gap="1.5" h="5" justify="space-between" minW="0" position="relative" w="full">
          {template.input !== 'direct' ? (
            <Tooltip content={handleTooltip} positioning={{ placement: 'right-end' }} showArrow>
              <Handle
                id={template.name}
                position={Position.Left}
                style={{ ...getWorkflowNodeHandleStyle(template.type, 'left'), left: -ROW_PADDING_X, top: '50%' }}
                type="target"
              />
            </Tooltip>
          ) : null}
          <Tooltip
            positioning={{ placement: 'top-start' }}
            content={
              <InputFieldTooltip
                description={getEffectiveWorkflowFieldDescription(instance, template)}
                isConnected={isConnected}
                isExposed={isExposed}
                label={label}
                template={template}
              />
            }
          >
            <Text
              color={isInvalid ? 'fg.error' : isConnected ? 'fg.muted' : 'fg'}
              fontSize="2xs"
              lineHeight="shorter"
              minW="0"
              truncate
            >
              {label}
              {template.required ? (
                <Text as="span" color="fg.error">
                  {' *'}
                </Text>
              ) : null}
            </Text>
          </Tooltip>
          <HStack flexShrink={0} gap="0" ml="auto">
            {canReset ? (
              <Tooltip content="Reset to default value">
                <IconButton
                  aria-label={`Reset ${label} to default value`}
                  className="nodrag"
                  color="fg.subtle"
                  size="2xs"
                  title="Reset to default value"
                  variant="ghost"
                  onClick={() =>
                    editGraph({
                      fieldName: template.name,
                      nodeId: node.id,
                      type: 'setFieldValue',
                      value: cloneWorkflowFieldDefault(template),
                    })
                  }
                >
                  <Icon as={RotateCcwIcon} boxSize="3" />
                </IconButton>
              </Tooltip>
            ) : null}
            <FieldDescriptionPopover
              description={instance?.description}
              fieldName={template.name}
              nodeId={node.id}
              templateDescription={template.description}
            />
            {isExposableField(template) ? (
              <IconButton
                aria-label={isExposed ? `Remove ${label} from Linear UI` : `Expose ${label} in Linear UI`}
                className="nodrag"
                color={isExposed ? 'accent.solid' : 'fg.subtle'}
                size="2xs"
                title={isExposed ? 'Remove from Linear UI form' : 'Expose in Linear UI form'}
                variant="ghost"
                onClick={() => editGraph({ fieldIdentifier, type: isExposed ? 'unexposeField' : 'exposeField' })}
              >
                <Icon as={isExposed ? PinOffIcon : PinIcon} boxSize="3" />
              </IconButton>
            ) : null}
          </HStack>
        </HStack>
        {showsControl ? (
          <Box mt="0.5" w="full">
            <WorkflowFieldInput
              id={`${node.id}-${template.name}-value`}
              invalid={isInvalid}
              nodeId={node.id}
              seedMode={getWorkflowFieldSeedMode(instance)}
              template={template}
              value={instance?.value}
              onChange={(value) =>
                editGraph({ fieldName: template.name, nodeId: node.id, type: 'setFieldValue', value })
              }
              onSeedModeChange={(seedMode) =>
                editGraph({ fieldName: template.name, nodeId: node.id, seedMode, type: 'setFieldSeedMode' })
              }
            />
          </Box>
        ) : isConnected && isSeedInputField(template) ? (
          <ProvidedByConnectionNote />
        ) : null}
        {invalidReason ? <Field.ErrorText fontSize="2xs">{invalidReason}</Field.ErrorText> : null}
      </Field.Root>
    </Box>
  );
};

const OutputFieldRow = ({
  isSkeleton,
  latestResult,
  template,
}: {
  isSkeleton: boolean;
  latestResult: unknown;
  template: FieldOutputTemplate;
}) => {
  const handleTooltip = getHandleTypeTooltip(template.type);
  const value = latestResult === undefined ? null : formatOutputFieldValue(latestResult, template.name);

  return (
    <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
      <Flex align="center" h="5" justify="flex-end" position="relative">
        <Tooltip content={handleTooltip} positioning={{ placement: 'left-start' }} showArrow>
          <Handle
            id={template.name}
            position={Position.Right}
            style={{ ...getWorkflowNodeHandleStyle(template.type, 'right'), right: -ROW_PADDING_X, top: '50%' }}
            type="source"
          />
        </Tooltip>
        {isSkeleton ? (
          <Flex justify="flex-end" w="full">
            <SkeletonBar w="40%" />
          </Flex>
        ) : (
          <HStack gap="1.5" justify="flex-end" minW="0" w="full">
            {value ? (
              <Tooltip content={value.full} positioning={{ placement: 'top-end' }}>
                <Text
                  color="fg.subtle"
                  fontSize="2xs"
                  lineHeight="shorter"
                  minW="0"
                  overflow="hidden"
                  textOverflow="ellipsis"
                  whiteSpace="nowrap"
                >
                  {value.short}
                </Text>
              </Tooltip>
            ) : null}
            <Box flexShrink={0} maxW={value ? '60%' : 'full'} textAlign="end">
              <Tooltip content={<OutputFieldTooltip template={template} />} positioning={{ placement: 'top-end' }}>
                <MiddleTruncate
                  as="span"
                  color="fg.muted"
                  fontSize="2xs"
                  justifyContent="flex-end"
                  lineHeight="shorter"
                  maxW="full"
                  text={template.title}
                />
              </Tooltip>
            </Box>
          </HStack>
        )}
      </Flex>
    </Box>
  );
};

const OutputScopeHeader = ({ scope }: { scope: 'iteration' | 'final' }) => {
  const { t } = useTranslation();

  return (
    <Box px={WORKFLOW_NODE_DENSITY.rowPaddingX} py={WORKFLOW_NODE_DENSITY.rowPaddingY}>
      <Text color="fg.subtle" fontSize="2xs" fontWeight="600" textAlign="end">
        {scope === 'iteration' ? t('nodes.iterationOutputs') : t('nodes.finalOutputs')}
      </Text>
    </Box>
  );
};

/** Keeps every handle mounted (invisible) so edges stay attached when rows are not rendered. */
const HiddenHandles = ({
  inputTemplates,
  outputTemplates,
}: {
  inputTemplates: FieldInputTemplate[];
  outputTemplates: FieldOutputTemplate[];
}) => (
  <Box position="relative" h="0">
    {outputTemplates.map((outputTemplate) => (
      <Handle
        key={outputTemplate.name}
        id={outputTemplate.name}
        position={Position.Right}
        style={{ ...getWorkflowNodeHandleStyle(outputTemplate.type, 'right'), opacity: 0, right: 0, top: -14 }}
        type="source"
      />
    ))}
    {inputTemplates.map((inputTemplate) =>
      inputTemplate.input !== 'direct' ? (
        <Handle
          key={inputTemplate.name}
          id={inputTemplate.name}
          position={Position.Left}
          style={{ ...getWorkflowNodeHandleStyle(inputTemplate.type, 'left'), left: 0, opacity: 0, top: -14 }}
          type="target"
        />
      ) : null
    )}
  </Box>
);

const CompactHiddenHandles = ({
  connectedSourceHandles,
  connectedTargetHandles,
  inputTemplates,
  outputTemplates,
}: {
  connectedSourceHandles: string[];
  connectedTargetHandles: string[];
  inputTemplates: FieldInputTemplate[];
  outputTemplates: FieldOutputTemplate[];
}) => {
  const connectedSources = new Set(connectedSourceHandles);
  const connectedTargets = new Set(connectedTargetHandles);

  return (
    <HiddenHandles
      inputTemplates={inputTemplates.filter((inputTemplate) => connectedTargets.has(inputTemplate.name))}
      outputTemplates={outputTemplates.filter((outputTemplate) => connectedSources.has(outputTemplate.name))}
    />
  );
};

const CompactNodeBody = ({ inputCount, outputCount }: { inputCount: number; outputCount: number }) => (
  <Flex
    align="center"
    color="fg.muted"
    fontSize="2xs"
    gap="2"
    px={WORKFLOW_NODE_DENSITY.rowPaddingX}
    {...getWorkflowNodeBodyProps()}
  >
    <Text>
      {inputCount} input{inputCount === 1 ? '' : 's'}
    </Text>
    <Text>·</Text>
    <Text>
      {outputCount} output{outputCount === 1 ? '' : 's'}
    </Text>
    <Text ms="auto">Select for fields</Text>
  </Flex>
);

const CompactInvocationNode = ({ data, selected }: NodeProps<InvocationFlowNodeType>) => {
  const node = data.documentNode;
  const templateView = data.template;
  const execution = useWorkflowNodeExecutionState(node.id);
  const inputTemplates = templateView?.inputTemplates ?? [];
  const outputTemplates = templateView?.outputTemplates ?? [];
  const title = node.data.label || templateView?.template.title || node.data.type;
  const isOutdated = templateView ? getNodeUpdateStatus(node, templateView.template) !== 'current' : false;

  return (
    <NodeShell
      isMissing={!templateView}
      isOutdated={isOutdated}
      isRunning={execution?.status === 'running'}
      outcome={getExecutionOutcome(execution)}
      selected={selected ?? false}
    >
      <Flex {...getWorkflowNodeHeaderProps()}>
        <MiddleTruncate fontSize="sm" fontWeight="700" minW="0" text={title} />
        <BatchGroupSuffix node={node} />
        <Box flex="1" />
        <NodeOutcomeIcon execution={execution} node={node} />
        {templateView && isOutdated ? <NodeUpdateIcon node={node} template={templateView.template} /> : null}
      </Flex>
      {templateView ? (
        <CompactNodeBody inputCount={inputTemplates.length} outputCount={outputTemplates.length} />
      ) : (
        <Text color="fg.subtle" fontSize="2xs" px="3" {...getWorkflowNodeBodyProps()} py="2">
          Unknown node type. Select for details.
        </Text>
      )}
      <CompactHiddenHandles
        connectedSourceHandles={data.connectedSourceHandles}
        connectedTargetHandles={data.connectedTargetHandles}
        inputTemplates={inputTemplates}
        outputTemplates={outputTemplates}
      />
    </NodeShell>
  );
};

const ExpandedInvocationNode = ({ data, selected }: NodeProps<InvocationFlowNodeType>) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const isZoomedOut = useIsZoomedOut();
  const node = data.documentNode;
  const templateView = data.template;
  const execution = useWorkflowNodeExecutionState(node.id);

  if (!templateView) {
    return (
      <NodeShell isMissing selected={selected ?? false}>
        <HStack gap="1.5" p="3">
          <Icon as={TriangleAlertIcon} boxSize="3.5" color="red.solid" />
          <Stack gap="0">
            <Text fontWeight="700">{node.data.label || node.data.type}</Text>
            <Text color="fg.subtle" fontSize="2xs">
              Unknown node type "{node.data.type}". It cannot run on this backend.
            </Text>
          </Stack>
        </HStack>
      </NodeShell>
    );
  }

  const template = templateView.template;
  const isOutdated = getNodeUpdateStatus(node, template) !== 'current';
  const connectedFieldNames = new Set(data.connectedTargetHandles);
  const exposedFieldNames = new Set(data.exposedFieldNames);
  const inputTemplates = templateView.inputTemplates;
  const outputTemplates = templateView.outputTemplates;
  const outputTemplatesByName = new Map(outputTemplates.map((template) => [template.name, template]));
  const outputRows = getOutputFieldRows(getOutputFieldNamesByScope(outputTemplates));
  const isOpen = node.data.isOpen;
  const isRunning = execution?.status === 'running';
  const isMissingRequiredInput = hasMissingRequiredInputs(node, inputTemplates, connectedFieldNames);
  const isCompact = data.isCompact && !selected;
  const withFooter = !isZoomedOut && templateView.isExecutable && templateView.hasImageOutput;
  const withOutputPreview = Boolean(execution?.outputImageUrl);
  const latestResult = execution?.latestOutput;

  return (
    <NodeShell
      hasMissingRequiredInput={isMissingRequiredInput}
      isOutdated={isOutdated}
      isRunning={isRunning}
      outcome={getExecutionOutcome(execution)}
      selected={selected ?? false}
    >
      {/* The collapse chevron carries its own hit padding, so the header pulls its start padding in. */}
      <Flex {...getWorkflowNodeHeaderProps({ roundedBottom: !isOpen })} gap="1" ps="1">
        <IconButton
          aria-label={isOpen ? 'Collapse node' : 'Expand node'}
          className="nodrag"
          size="2xs"
          variant="ghost"
          onClick={() => editGraph({ isOpen: !isOpen, nodeId: node.id, type: 'setNodeIsOpen' })}
        >
          <Icon as={isOpen ? ChevronDownIcon : ChevronRightIcon} boxSize="3.5" />
        </IconButton>
        {isZoomedOut ? (
          <MiddleTruncate fontSize="sm" fontWeight="700" minW="0" text={node.data.label || template.title} />
        ) : (
          <>
            <NodeTitle node={node} title={node.data.label || template.title} />
            <BatchGroupSuffix node={node} />
            <Box flex="1" />
            <NodeOutcomeIcon execution={execution} node={node} />
            {isOutdated ? <NodeUpdateIcon node={node} template={template} /> : null}
            <NodeInfoIcon node={node} template={template} />
          </>
        )}
      </Flex>
      <NodeProgressStrip execution={execution} />
      {isOpen && isCompact ? (
        <>
          <CompactNodeBody inputCount={inputTemplates.length} outputCount={outputTemplates.length} />
          <HiddenHandles inputTemplates={inputTemplates} outputTemplates={outputTemplates} />
        </>
      ) : isOpen ? (
        <Box {...getWorkflowNodeBodyProps({ roundedBottom: !withFooter && !withOutputPreview })}>
          {outputRows.map((row) =>
            row.type === 'header' ? (
              <OutputScopeHeader key={`output-scope-${row.scope}`} scope={row.scope} />
            ) : (
              <OutputFieldRow
                key={row.fieldName}
                isSkeleton={isZoomedOut}
                latestResult={latestResult}
                template={outputTemplatesByName.get(row.fieldName) as FieldOutputTemplate}
              />
            )
          )}
          {inputTemplates.map((inputTemplate) => (
            <InputFieldRow
              key={inputTemplate.name}
              isConnected={connectedFieldNames.has(inputTemplate.name)}
              isExposed={exposedFieldNames.has(inputTemplate.name)}
              isSkeleton={isZoomedOut}
              node={node}
              template={inputTemplate}
            />
          ))}
          {shouldShowCallSavedWorkflowLoadingHint(node) ? (
            <Text color="fg.subtle" fontSize="2xs" px={WORKFLOW_NODE_DENSITY.rowPaddingX} py="1">
              {t('nodes.savedWorkflowDetailLoading')}
            </Text>
          ) : shouldShowCallSavedWorkflowNoExposedFieldsHint(node) ? (
            <Text color="fg.subtle" fontSize="2xs" px={WORKFLOW_NODE_DENSITY.rowPaddingX} py="1">
              {t('nodes.savedWorkflowNoExposedFields')}
            </Text>
          ) : null}
        </Box>
      ) : (
        <HiddenHandles inputTemplates={inputTemplates} outputTemplates={outputTemplates} />
      )}
      {isOpen && execution?.outputImageUrl ? (
        <NodeOutputPreview
          imageUrl={execution.outputImageUrl}
          isSkeleton={isZoomedOut}
          nodeId={node.id}
          roundedBottom={!withFooter}
        />
      ) : null}
      {isOpen && withFooter ? <NodeFooter canUseCache={data.canUseCache} node={node} /> : null}
    </NodeShell>
  );
};

const InvocationFlowNodeComponent = (props: NodeProps<InvocationFlowNodeType>) =>
  props.data.isCompact && !props.selected ? (
    <CompactInvocationNode {...props} />
  ) : (
    <ExpandedInvocationNode {...props} />
  );

export const InvocationFlowNode = memo(InvocationFlowNodeComponent);
