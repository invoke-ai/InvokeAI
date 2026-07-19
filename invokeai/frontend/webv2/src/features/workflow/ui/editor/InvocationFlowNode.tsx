/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  WorkflowInvocationNode,
} from '@features/workflow/contracts';
import type { WorkflowNodeExecutionState as NodeExecutionState } from '@features/workflow/ui/contracts';

import { Box, Checkbox, Field, Flex, HStack, Icon, IconButton, Image, Input, Stack, Text } from '@chakra-ui/react';
import { FieldDescriptionPopover } from '@features/workflow/ui/fields/FieldDescriptionPopover';
import { WorkflowFieldInput } from '@features/workflow/ui/fields/WorkflowFieldInput';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowNodeExecutionState } from '@features/workflow/ui/WorkflowUiContext';
import {
  cloneWorkflowFieldDefault,
  getFieldTypeColor,
  getFieldTypeLabel,
  getWorkflowFieldInvalidReason,
  isDirectInputField,
  isExposableField,
  isWorkflowFieldValueDefault,
  isModelFieldType,
} from '@features/workflow/utility';
import { Tooltip } from '@platform/ui';
import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  InfoIcon,
  PinIcon,
  PinOffIcon,
  RotateCcwIcon,
  TriangleAlertIcon,
} from 'lucide-react';
import { memo, useState, type ChangeEvent, type KeyboardEvent } from 'react';

import type { InvocationFlowNode as InvocationFlowNodeType, InvocationNodeTemplateView } from './flowAdapters';

import { getHandleTypeTooltip } from './handleTooltip';
import { getWorkflowNodeChromeProps } from './nodeChrome';

const NODE_WIDTH = '18rem';
const HANDLE_SIZE = 12;
type HandleSide = 'left' | 'right';
/** Row padding-x in px; inline handles sit in the label rows, so they pull back out past it. */
const ROW_PADDING_X = 12;
/** Below this viewport zoom, field content renders as skeleton bars (ComfyUI-style) for performance/readability. */
const CONTENT_VISIBILITY_ZOOM = 0.4;

const handleStyle = (type: FieldType, side: HandleSide): React.CSSProperties => {
  const color = getFieldTypeColor(type);
  const isFilled = type.cardinality === 'SINGLE';
  const isAngular = isModelFieldType(type) || type.batch;
  const centeredDiamondTransform = `translate(${side === 'left' ? '-' : ''}50%, -50%) rotate(45deg)`;

  return {
    background: isFilled ? color : 'var(--xy-background-color)',
    border: isFilled ? 'none' : `3px solid ${color}`,
    borderRadius: isAngular ? 3 : '50%',
    boxShadow: '0 0 0 1.5px var(--xy-background-color)',
    height: HANDLE_SIZE,
    transform: type.batch ? centeredDiamondTransform : undefined,
    width: HANDLE_SIZE,
  };
};

/** True while the viewport is zoomed out far enough that field content is unreadable noise. */
const useIsZoomedOut = (): boolean => useStore((state) => state.transform[2] < CONTENT_VISIBILITY_ZOOM);

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

const NodeShell = ({
  hasMissingRequiredInput,
  children,
  isMissing,
  isRunning,
  selected,
}: {
  hasMissingRequiredInput?: boolean;
  children: React.ReactNode;
  isMissing?: boolean;
  isRunning?: boolean;
  selected: boolean;
}) => {
  const isInvalid = isMissing || hasMissingRequiredInput;

  return (
    <Box
      bg="bg"
      fontSize="xs"
      rounded="lg"
      w={NODE_WIDTH}
      {...getWorkflowNodeChromeProps({ invalid: isInvalid, running: isRunning, selected })}
    >
      {children}
    </Box>
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
    <Text
      fontWeight="700"
      minW="0"
      title="Double-click to rename"
      truncate
      // Editing always starts from the displayed title: an unset label
      // prefills with the template title rather than an empty input.
      onDoubleClick={() => setDraftLabel(title)}
    >
      {title}
    </Text>
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

const OutputFieldTooltip = ({ template }: { template: FieldOutputTemplate }) => (
  <Stack gap="0.5" maxW="18rem">
    <Text fontWeight="700">{template.title}</Text>
    <Text color="fg.subtle">Field: {template.name}</Text>
    <Text color="fg.subtle">Type: {getFieldTypeLabel(template.type)}</Text>
    <Text color="fg.subtle">Output</Text>
    {template.description ? <Text>{template.description}</Text> : null}
  </Stack>
);

const NodeInfoTooltipContent = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => {
  const title = node.data.label ? `${node.data.label} (${template.title})` : template.title;
  const nodePack = node.data.nodePack || template.nodePack;

  return (
    <Stack gap="1" maxW="20rem">
      <Text fontWeight="700">{title}</Text>
      <Text color="fg.subtle">Type: {template.type}</Text>
      <Text color="fg.subtle">Node pack: {nodePack}</Text>
      <Text color="fg.subtle">Version: {node.data.version}</Text>
      <Text color="fg.subtle">Classification: {template.classification}</Text>
      <Text color="fg.subtle">Category: {template.category}</Text>
      {template.description ? <Text fontStyle="italic">{template.description}</Text> : null}
      {node.data.notes ? <Text>{node.data.notes}</Text> : null}
    </Stack>
  );
};

const NodeInfoButton = ({
  node,
  template,
}: {
  node: WorkflowInvocationNode;
  template: InvocationNodeTemplateView['template'];
}) => (
  <Tooltip
    content={<NodeInfoTooltipContent node={node} template={template} />}
    positioning={{ placement: 'top-end' }}
    showArrow
  >
    <IconButton
      aria-label={`Show details for ${node.data.label || template.title}`}
      className="nodrag"
      size="2xs"
      variant="ghost"
    >
      <Icon as={InfoIcon} boxSize="3.5" />
    </IconButton>
  </Tooltip>
);

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
      py="1"
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
      <Box px="3" py="1.5">
        <HStack gap="1.5" h="5" position="relative">
          {template.input !== 'direct' ? (
            <Tooltip content={handleTooltip} showArrow>
              <Handle
                id={template.name}
                position={Position.Left}
                style={{ ...handleStyle(template.type, 'left'), left: -ROW_PADDING_X, top: '50%' }}
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
    <Box px="3" py="1.5" w="full">
      <Field.Root gap="0" invalid={isInvalid} minW="0" w="full">
        {/* The handle lives inside the label row so it stays centered on the
            label even when the value control below grows the row. */}
        <HStack gap="1.5" h="5" justify="space-between" minW="0" position="relative" w="full">
          {template.input !== 'direct' ? (
            <Tooltip content={handleTooltip} positioning={{ placement: 'right-end' }} showArrow>
              <Handle
                id={template.name}
                position={Position.Left}
                style={{ ...handleStyle(template.type, 'left'), left: -ROW_PADDING_X, top: '50%' }}
                type="target"
              />
            </Tooltip>
          ) : null}
          <Tooltip
            positioning={{ placement: 'top-start' }}
            content={
              <InputFieldTooltip
                description={instance?.description || template.description}
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
              template={template}
              value={instance?.value}
              onChange={(value) =>
                editGraph({ fieldName: template.name, nodeId: node.id, type: 'setFieldValue', value })
              }
            />
          </Box>
        ) : null}
        {invalidReason ? <Field.ErrorText fontSize="2xs">{invalidReason}</Field.ErrorText> : null}
      </Field.Root>
    </Box>
  );
};

const OutputFieldRow = ({ isSkeleton, template }: { isSkeleton: boolean; template: FieldOutputTemplate }) => {
  const handleTooltip = getHandleTypeTooltip(template.type);

  return (
    <Box px="3" py="1">
      <Flex align="center" h="5" justify="flex-end" position="relative">
        <Tooltip content={handleTooltip} positioning={{ placement: 'left-start' }} showArrow>
          <Handle
            id={template.name}
            position={Position.Right}
            style={{ ...handleStyle(template.type, 'right'), right: -ROW_PADDING_X, top: '50%' }}
            type="source"
          />
        </Tooltip>
        {isSkeleton ? (
          <Flex justify="flex-end" w="full">
            <SkeletonBar w="40%" />
          </Flex>
        ) : (
          <Box textAlign="end">
            <Tooltip content={<OutputFieldTooltip template={template} />} positioning={{ placement: 'top-end' }}>
              <Text
                as="span"
                color="fg.muted"
                display="inline-block"
                fontSize="2xs"
                lineHeight="shorter"
                maxW="full"
                textAlign="end"
                truncate
              >
                {template.title}
              </Text>
            </Tooltip>
          </Box>
        )}
      </Flex>
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
        style={{ ...handleStyle(outputTemplate.type, 'right'), opacity: 0, right: 0, top: -14 }}
        type="source"
      />
    ))}
    {inputTemplates.map((inputTemplate) =>
      inputTemplate.input !== 'direct' ? (
        <Handle
          key={inputTemplate.name}
          id={inputTemplate.name}
          position={Position.Left}
          style={{ ...handleStyle(inputTemplate.type, 'left'), left: 0, opacity: 0, top: -14 }}
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
    bg="bg.muted"
    borderBottomRadius="lg"
    borderTopWidth="1px"
    borderColor="border.subtle"
    color="fg.subtle"
    fontSize="2xs"
    gap="2"
    px="3"
    py="2"
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
  const inputTemplates = templateView?.inputTemplates ?? [];
  const outputTemplates = templateView?.outputTemplates ?? [];
  const title = node.data.label || templateView?.template.title || node.data.type;

  return (
    <NodeShell isMissing={!templateView} selected={selected ?? false}>
      <Flex
        align="center"
        bg="bg.subtle"
        borderBottomWidth="1px"
        borderColor="border.subtle"
        borderTopRadius="lg"
        gap="1"
        px="3"
        py="2"
      >
        <Text fontSize="sm" fontWeight="700" minW="0" title={title} truncate>
          {title}
        </Text>
      </Flex>
      {templateView ? (
        <CompactNodeBody inputCount={inputTemplates.length} outputCount={outputTemplates.length} />
      ) : (
        <Text bg="bg.muted" borderBottomRadius="lg" color="fg.subtle" fontSize="2xs" px="3" py="2">
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
  const connectedFieldNames = new Set(data.connectedTargetHandles);
  const exposedFieldNames = new Set(data.exposedFieldNames);
  const inputTemplates = templateView.inputTemplates;
  const outputTemplates = templateView.outputTemplates;
  const isOpen = node.data.isOpen;
  const isRunning = execution?.status === 'running';
  const isMissingRequiredInput = hasMissingRequiredInputs(node, Object.values(template.inputs), connectedFieldNames);
  const isCompact = data.isCompact && !selected;
  const withFooter = !isZoomedOut && templateView.isExecutable && templateView.hasImageOutput;
  const withOutputPreview = Boolean(execution?.outputImageUrl);

  return (
    <NodeShell hasMissingRequiredInput={isMissingRequiredInput} isRunning={isRunning} selected={selected ?? false}>
      <Flex
        align="center"
        bg="bg.subtle"
        borderBottomWidth={isOpen ? '1px' : '0'}
        borderColor="border.subtle"
        borderTopRadius="lg"
        borderBottomRadius={isOpen ? 'none' : 'lg'}
        gap="1"
        ps="1.5"
        pe="2"
        py="1.5"
      >
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
          <Text fontSize="sm" fontWeight="700" minW="0" truncate>
            {node.data.label || template.title}
          </Text>
        ) : (
          <>
            <NodeTitle node={node} title={node.data.label || template.title} />
            <Box flex="1" />
            <NodeInfoButton node={node} template={template} />
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
        <Box bg="bg.muted" borderBottomRadius={withFooter || withOutputPreview ? 'none' : 'lg'} py="1">
          {outputTemplates.map((outputTemplate) => (
            <OutputFieldRow key={outputTemplate.name} isSkeleton={isZoomedOut} template={outputTemplate} />
          ))}
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
        </Box>
      ) : (
        <HiddenHandles inputTemplates={inputTemplates} outputTemplates={outputTemplates} />
      )}
      {isOpen && execution?.outputImageUrl ? (
        <Box borderBottomRadius={withFooter ? 'none' : 'lg'} borderColor="border.subtle" borderTopWidth="1px" p="1.5">
          {isZoomedOut ? (
            <SkeletonBar h="6rem" w="full" />
          ) : (
            <Image
              alt="Latest output of this node"
              draggable={false}
              maxH="10rem"
              mx="auto"
              objectFit="contain"
              rounded="sm"
              src={execution.outputImageUrl}
            />
          )}
        </Box>
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
