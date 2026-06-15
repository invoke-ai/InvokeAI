import { Badge, Box, Flex, HStack, Icon, Image, Input, Stack, Text, IconButton } from '@chakra-ui/react';
import { Handle, Position, useStore, type NodeProps } from '@xyflow/react';
import { ChevronDownIcon, ChevronRightIcon, PinIcon, PinOffIcon, TriangleAlertIcon } from 'lucide-react';
import { memo, useState, type ChangeEvent, type KeyboardEvent } from 'react';

import { useNodeExecutionState, type NodeExecutionState } from '../../../backend/nodeExecutionStore';
import { Tooltip } from '../../../components/ui/Tooltip';
import { useWorkbenchDispatch } from '../../../WorkbenchContext';
import { getFieldTypeColor, getFieldTypeLabel, isDirectInputField, isExposableField } from '../../../workflows/fields';
import { useInvocationTemplatesSnapshot } from '../../../workflows/templates';
import type { FieldInputTemplate, FieldOutputTemplate, WorkflowInvocationNode } from '../../../workflows/types';
import { FieldDescriptionPopover } from '../fields/FieldDescriptionPopover';
import { WorkflowFieldInput } from '../fields/WorkflowFieldInput';
import type { InvocationFlowNode as InvocationFlowNodeType } from './flowAdapters';

const NODE_WIDTH = '18rem';
const HANDLE_SIZE = 10;
/** Row padding-x in px; inline handles sit in the label rows, so they pull back out past it. */
const ROW_PADDING_X = 12;
/** Below this viewport zoom, field content renders as skeleton bars (ComfyUI-style) for performance/readability. */
const CONTENT_VISIBILITY_ZOOM = 0.4;

const handleStyle = (color: string): React.CSSProperties => ({
  background: color,
  border: 'none',
  height: HANDLE_SIZE,
  width: HANDLE_SIZE,
});

const sortByUiOrder = <T extends { uiOrder?: number | null }>(templates: T[]): T[] =>
  [...templates].sort((a, b) => (a.uiOrder ?? Number.MAX_SAFE_INTEGER) - (b.uiOrder ?? Number.MAX_SAFE_INTEGER));

/** True while the viewport is zoomed out far enough that field content is unreadable noise. */
const useIsZoomedOut = (): boolean => useStore((state) => state.transform[2] < CONTENT_VISIBILITY_ZOOM);

/** Static placeholder bar standing in for text/controls at far zoom. No animation — there may be hundreds. */
const SkeletonBar = ({ h = '2', w }: { h?: string; w?: string }) => <Box bg="bg.emphasized" h={h} rounded="sm" w={w} />;

const NodeShell = ({
  children,
  isMissing,
  isRunning,
  selected,
}: {
  children: React.ReactNode;
  isMissing?: boolean;
  isRunning?: boolean;
  selected: boolean;
}) => (
  <Box
    bg="bg"
    borderColor={isMissing ? 'red.solid' : isRunning ? 'brand.solid' : selected ? 'accent.solid' : 'border.emphasized'}
    borderWidth="1px"
    fontSize="xs"
    rounded="lg"
    shadow={isRunning ? '0 0 10px {colors.brand.solid/50}' : selected ? 'md' : 'sm'}
    w={NODE_WIDTH}
  >
    {children}
  </Box>
);

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
        transition="width 0.2s ease"
        w={execution.progress === null ? 'full' : `${Math.round(execution.progress * 100)}%`}
        {...(execution.progress === null
          ? { animationDuration: '1.2s', animationIterationCount: 'infinite', animationName: 'pulse' }
          : {})}
      />
    </Box>
  );
};

const NodeTitle = ({ node }: { node: WorkflowInvocationNode }) => {
  const dispatch = useWorkbenchDispatch();
  const { templates } = useInvocationTemplatesSnapshot();
  const [draftLabel, setDraftLabel] = useState<string | null>(null);
  const title = node.data.label || templates[node.data.type]?.title || node.data.type;

  if (draftLabel !== null) {
    return (
      <Input
        autoFocus
        aria-label="Node label"
        className="nodrag"
        size="2xs"
        value={draftLabel}
        onBlur={() => {
          dispatch({
            action: { label: draftLabel.trim(), nodeId: node.id, type: 'setNodeLabel' },
            type: 'applyProjectGraphAction',
          });
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
  const dispatch = useWorkbenchDispatch();
  const instance = node.data.inputs[template.name];
  const fieldIdentifier = { fieldName: template.name, nodeId: node.id };
  const showsControl = !isConnected && isDirectInputField(template);
  const label = instance?.label || template.title;

  if (isSkeleton) {
    return (
      <Box px="3" py="1.5">
        <HStack gap="1.5" h="5" position="relative">
          {template.input !== 'direct' ? (
            <Handle
              id={template.name}
              position={Position.Left}
              style={{ ...handleStyle(getFieldTypeColor(template.type)), left: -ROW_PADDING_X, top: '50%' }}
              type="target"
            />
          ) : null}
          <SkeletonBar w="55%" />
        </HStack>
        {showsControl ? <SkeletonBar h="6" w="full" /> : null}
      </Box>
    );
  }

  return (
    <Box px="3" py="1.5">
      {/* The handle lives inside the label row so it stays centered on the
          label even when the value control below grows the row. */}
      <HStack gap="1.5" justify="space-between" position="relative">
        {template.input !== 'direct' ? (
          <Handle
            id={template.name}
            position={Position.Left}
            style={{ ...handleStyle(getFieldTypeColor(template.type)), left: -ROW_PADDING_X, top: '50%' }}
            type="target"
          />
        ) : null}
        <Tooltip
          content={`${instance?.description || template.description || label} — ${getFieldTypeLabel(template.type)}`}
        >
          <Text color={isConnected ? 'fg.muted' : 'fg'} fontSize="2xs" minW="0" truncate>
            {label}
            {template.required ? (
              <Text as="span" color="fg.error">
                {' *'}
              </Text>
            ) : null}
          </Text>
        </Tooltip>
        <HStack flexShrink={0} gap="0">
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
              onClick={() =>
                dispatch({
                  action: { fieldIdentifier, type: isExposed ? 'unexposeField' : 'exposeField' },
                  type: 'applyProjectGraphAction',
                })
              }
            >
              <Icon as={isExposed ? PinOffIcon : PinIcon} boxSize="3" />
            </IconButton>
          ) : null}
        </HStack>
      </HStack>
      {showsControl ? (
        <Box mt="1">
          <WorkflowFieldInput
            template={template}
            value={instance?.value}
            onChange={(value) =>
              dispatch({
                action: { fieldName: template.name, nodeId: node.id, type: 'setFieldValue', value },
                type: 'applyProjectGraphAction',
              })
            }
          />
        </Box>
      ) : null}
    </Box>
  );
};

const OutputFieldRow = ({ isSkeleton, template }: { isSkeleton: boolean; template: FieldOutputTemplate }) => (
  <Box px="3" py="1">
    <Box position="relative">
      <Handle
        id={template.name}
        position={Position.Right}
        style={{ ...handleStyle(getFieldTypeColor(template.type)), right: -ROW_PADDING_X, top: '50%' }}
        type="source"
      />
      {isSkeleton ? (
        <Flex h="4" justify="flex-end">
          <SkeletonBar w="40%" />
        </Flex>
      ) : (
        <Tooltip content={`${template.description || template.title} — ${getFieldTypeLabel(template.type)}`}>
          <Text color="fg.muted" fontSize="2xs" textAlign="end" truncate>
            {template.title}
          </Text>
        </Tooltip>
      )}
    </Box>
  </Box>
);

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
        style={{ ...handleStyle(getFieldTypeColor(outputTemplate.type)), opacity: 0, right: 0, top: -14 }}
        type="source"
      />
    ))}
    {inputTemplates.map((inputTemplate) =>
      inputTemplate.input !== 'direct' ? (
        <Handle
          key={inputTemplate.name}
          id={inputTemplate.name}
          position={Position.Left}
          style={{ ...handleStyle(getFieldTypeColor(inputTemplate.type)), left: 0, opacity: 0, top: -14 }}
          type="target"
        />
      ) : null
    )}
  </Box>
);

const InvocationFlowNodeComponent = ({ data, selected }: NodeProps<InvocationFlowNodeType>) => {
  const dispatch = useWorkbenchDispatch();
  const { templates } = useInvocationTemplatesSnapshot();
  const isZoomedOut = useIsZoomedOut();
  const node = data.documentNode;
  const template = templates[node.data.type];
  const execution = useNodeExecutionState(node.id);

  if (!template) {
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

  const connectedFieldNames = new Set(data.connectedTargetHandles);
  const exposedFieldNames = new Set(data.exposedFieldNames);
  const inputTemplates = sortByUiOrder(Object.values(template.inputs).filter((input) => !input.uiHidden));
  const outputTemplates = Object.values(template.outputs);
  const isOpen = node.data.isOpen;
  const isRunning = execution?.status === 'running';

  return (
    <NodeShell isRunning={isRunning} selected={selected ?? false}>
      <Flex
        align="center"
        bg="bg.subtle"
        borderBottomWidth={isOpen ? '1px' : '0'}
        borderColor="border.subtle"
        borderTopRadius="lg"
        borderBottomRadius={isOpen ? 'none' : 'lg'}
        gap="1"
        px="2"
        py="1.5"
      >
        <IconButton
          aria-label={isOpen ? 'Collapse node' : 'Expand node'}
          className="nodrag"
          size="2xs"
          variant="ghost"
          onClick={() =>
            dispatch({
              action: { isOpen: !isOpen, nodeId: node.id, type: 'setNodeIsOpen' },
              type: 'applyProjectGraphAction',
            })
          }
        >
          <Icon as={isOpen ? ChevronDownIcon : ChevronRightIcon} boxSize="3.5" />
        </IconButton>
        {isZoomedOut ? (
          <Text fontSize="sm" fontWeight="700" minW="0" truncate>
            {node.data.label || template.title}
          </Text>
        ) : (
          <>
            <NodeTitle node={node} />
            <Box flex="1" />
            <Tooltip content={template.description || template.type}>
              <Badge size="xs" fontFamily="mono">
                {template.type}
              </Badge>
            </Tooltip>
          </>
        )}
      </Flex>
      <NodeProgressStrip execution={execution} />
      {isOpen ? (
        <Box py="1" bg="bg.muted">
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
        <Box borderColor="border.subtle" borderTopWidth="1px" p="1.5">
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
    </NodeShell>
  );
};

export const InvocationFlowNode = memo(InvocationFlowNodeComponent);
