import type {
  ContainerFormElement,
  NodeFieldFormElement,
  ProjectGraphState,
  WorkflowFormElement,
} from '@workbench/workflows/types';

import { Box, HStack, Icon, Input, Menu, Portal, Separator, Stack, Text, Textarea } from '@chakra-ui/react';
import { Button, IconButton } from '@workbench/components/ui/Button';
import { requestNodeSelection, workflowSelectionStore } from '@workbench/widgets/workflow/editor/selectionStore';
import { FieldDescriptionPopover } from '@workbench/widgets/workflow/fields/FieldDescriptionPopover';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { getResolvedWorkflowEdges } from '@workbench/workflows/connectors';
import { getFormChildren } from '@workbench/workflows/document';
import { getWorkflowFieldInvalidReason } from '@workbench/workflows/fields';
import { useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import { isInvocationNode } from '@workbench/workflows/types';
import {
  Columns2Icon,
  CrosshairIcon,
  GripVerticalIcon,
  HeadingIcon,
  InfoIcon,
  MinusIcon,
  PlusIcon,
  Rows2Icon,
  TextIcon,
  XIcon,
} from 'lucide-react';
import { createContext, use, useMemo, useState, type ChangeEvent, type DragEvent, type ReactNode } from 'react';

import { NodeFieldControl, useNodeFieldBinding } from './NodeFieldControl';

/**
 * The form builder: edit mode of the Linear UI. Every element renders as a
 * card with its own title bar — type label on the left, actions on the right,
 * content below — mirroring the legacy builder. Cards reorder and reparent by
 * dragging their title bar (drop indicators above/below, containers accept
 * drops into their body). All edits go through the project graph document
 * reducer.
 */

interface BuilderDndContextValue {
  draggingElementId: string | null;
  setDraggingElementId: (elementId: string | null) => void;
}

const BuilderDndContext = createContext<BuilderDndContextValue>({
  draggingElementId: null,
  setDraggingElementId: () => undefined,
});

type DropEdge = 'above' | 'below';

/** A builder card: typed title bar (drag handle + actions) over the element's content. */
const BuilderCard = ({
  children,
  element,
  extraActions,
  index,
  isHovered,
  isInvalid,
  isSelected,
  parentId,
  title,
}: {
  children: ReactNode;
  element: WorkflowFormElement;
  extraActions?: ReactNode;
  index: number;
  isHovered?: boolean;
  isInvalid?: boolean;
  isSelected?: boolean;
  parentId: string;
  title: string;
}) => {
  const dispatch = useWorkbenchDispatch();
  const { draggingElementId, setDraggingElementId } = use(BuilderDndContext);
  const [isDragArmed, setIsDragArmed] = useState(false);
  const [dropEdge, setDropEdge] = useState<DropEdge | null>(null);
  const isDropTarget = draggingElementId !== null && draggingElementId !== element.id;

  const onDragOver = (event: DragEvent<HTMLDivElement>) => {
    if (!isDropTarget) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const bounds = event.currentTarget.getBoundingClientRect();

    setDropEdge(event.clientY < bounds.top + bounds.height / 2 ? 'above' : 'below');
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    if (!isDropTarget || !draggingElementId || !dropEdge) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    dispatch({
      action: {
        elementId: draggingElementId,
        index: dropEdge === 'above' ? index : index + 1,
        parentId,
        type: 'moveFormElementTo',
      },
      type: 'applyProjectGraphAction',
    });
    setDropEdge(null);
  };

  return (
    <Box
      draggable={isDragArmed}
      flex="1"
      minW="0"
      opacity={draggingElementId === element.id ? 0.4 : 1}
      position="relative"
      onDragEnd={() => {
        setDraggingElementId(null);
        setIsDragArmed(false);
      }}
      onDragLeave={() => setDropEdge(null)}
      onDragOver={onDragOver}
      onDragStart={(event: DragEvent<HTMLDivElement>) => {
        event.stopPropagation();
        event.dataTransfer.effectAllowed = 'move';
        setDraggingElementId(element.id);
      }}
      onDrop={onDrop}
    >
      {dropEdge ? (
        <Box
          bg="accent.solid"
          h="2px"
          left="0"
          pointerEvents="none"
          position="absolute"
          right="0"
          rounded="full"
          zIndex="1"
          {...(dropEdge === 'above' ? { top: '-1px' } : { bottom: '-1px' })}
        />
      ) : null}
      <Box
        borderColor={isInvalid ? 'red.solid' : isHovered || isSelected ? 'accent.solid' : 'border.subtle'}
        borderWidth="1px"
        overflow="hidden"
        position="relative"
        rounded="md"
        shadow={
          isInvalid
            ? '0 0 0 1px {colors.red.solid/45}'
            : isHovered || isSelected
              ? '0 0 0 1px {colors.accent.solid/45}'
              : undefined
        }
        transition="border-color 0.12s ease, box-shadow 0.12s ease"
        _before={
          isHovered
            ? {
                bg: 'accent.solid/10',
                content: '""',
                inset: '0',
                pointerEvents: 'none',
                position: 'absolute',
                zIndex: '1',
              }
            : undefined
        }
      >
        <HStack
          bg="bg.muted"
          borderBottomWidth="1px"
          borderColor="border.subtle"
          cursor="grab"
          gap="1"
          px="1.5"
          py="0.5"
          position="relative"
          zIndex="2"
          _active={{ cursor: 'grabbing' }}
          onPointerDown={() => setIsDragArmed(true)}
          onPointerUp={() => setIsDragArmed(false)}
        >
          <Icon as={GripVerticalIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
          <Text color="fg.muted" fontSize="2xs" fontWeight="600" minW="0" truncate>
            {title}
          </Text>
          <Box flex="1" />
          <HStack flexShrink={0} gap="0" onPointerDown={(event) => event.stopPropagation()}>
            {extraActions}
            <IconButton
              aria-label="Remove from form"
              size="2xs"
              variant="ghost"
              onClick={() =>
                dispatch({
                  action: { elementId: element.id, type: 'removeFormElement' },
                  type: 'applyProjectGraphAction',
                })
              }
            >
              <Icon as={XIcon} boxSize="3" />
            </IconButton>
          </HStack>
        </HStack>
        <Box p="2" position="relative" zIndex="2">
          {children}
        </Box>
      </Box>
    </Box>
  );
};

/** Drop zone covering a container's body, appending at the end. Doubles as the empty-container hint. */
const ContainerDropZone = ({ container, isEmpty }: { container: ContainerFormElement; isEmpty: boolean }) => {
  const dispatch = useWorkbenchDispatch();
  const { draggingElementId } = use(BuilderDndContext);
  const [isActive, setIsActive] = useState(false);
  const canDrop = draggingElementId !== null && draggingElementId !== container.id;

  if (!canDrop && !isEmpty) {
    return null;
  }

  return (
    <Box
      alignSelf="stretch"
      borderColor={isActive ? 'accent.solid' : 'border.subtle'}
      borderStyle="dashed"
      borderWidth="1px"
      color="fg.subtle"
      flex={isEmpty ? '1' : undefined}
      fontSize="2xs"
      px="2"
      py="1.5"
      rounded="md"
      textAlign="center"
      onDragLeave={() => setIsActive(false)}
      onDragOver={(event: DragEvent<HTMLDivElement>) => {
        if (canDrop) {
          event.preventDefault();
          event.stopPropagation();
          setIsActive(true);
        }
      }}
      onDrop={(event: DragEvent<HTMLDivElement>) => {
        if (!canDrop || !draggingElementId) {
          return;
        }

        event.preventDefault();
        event.stopPropagation();
        dispatch({
          action: {
            elementId: draggingElementId,
            index: container.data.children.length,
            parentId: container.id,
            type: 'moveFormElementTo',
          },
          type: 'applyProjectGraphAction',
        });
        setIsActive(false);
      }}
    >
      {canDrop ? 'Drop here' : 'Empty container — drag elements here'}
    </Box>
  );
};

/** The shared description popover, bound through the form element. */
const FieldDescriptionAction = ({
  element,
  projectGraph,
}: {
  element: NodeFieldFormElement;
  projectGraph: ProjectGraphState;
}) => {
  const { fieldName, instance, nodeId, template } = useNodeFieldBinding(element, projectGraph);

  if (!template) {
    return null;
  }

  return (
    <FieldDescriptionPopover
      description={instance?.description}
      fieldName={fieldName}
      nodeId={nodeId}
      templateDescription={template.description}
    />
  );
};

const BuilderElement = ({
  element,
  index,
  hoveredNodeId,
  parentId,
  projectGraph,
  invalidElementIds,
  selectedNodeIds,
}: {
  element: WorkflowFormElement;
  index: number;
  hoveredNodeId: string | null;
  invalidElementIds: Set<string>;
  parentId: string;
  projectGraph: ProjectGraphState;
  selectedNodeIds: Set<string>;
}) => {
  const dispatch = useWorkbenchDispatch();

  switch (element.type) {
    case 'container': {
      const isRow = element.data.layout === 'row';

      return (
        <BuilderCard
          element={element}
          extraActions={
            <IconButton
              aria-label={isRow ? 'Switch container to column layout' : 'Switch container to row layout'}
              size="2xs"
              title={isRow ? 'Switch to column layout' : 'Switch to row layout'}
              variant="ghost"
              onClick={() =>
                dispatch({
                  action: { elementId: element.id, layout: isRow ? 'column' : 'row', type: 'setContainerLayout' },
                  type: 'applyProjectGraphAction',
                })
              }
            >
              <Icon as={isRow ? Rows2Icon : Columns2Icon} boxSize="3" />
            </IconButton>
          }
          index={index}
          parentId={parentId}
          title={`Container (${element.data.layout} layout)`}
        >
          <Stack align={isRow ? 'stretch' : undefined} direction={isRow ? 'row' : 'column'} gap="2" w="full">
            {getFormChildren(projectGraph.form, element.id).map((child, childIndex) => (
              <BuilderElement
                key={child.id}
                element={child}
                hoveredNodeId={hoveredNodeId}
                invalidElementIds={invalidElementIds}
                index={childIndex}
                parentId={element.id}
                projectGraph={projectGraph}
                selectedNodeIds={selectedNodeIds}
              />
            ))}
            <ContainerDropZone container={element} isEmpty={element.data.children.length === 0} />
          </Stack>
        </BuilderCard>
      );
    }
    case 'node-field': {
      return (
        <BuilderCard
          element={element}
          extraActions={
            <>
              <IconButton
                aria-label="Zoom to node in editor"
                size="2xs"
                title="Zoom to node in the Workflow editor"
                variant="ghost"
                onClick={() => {
                  dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: 'workflow' });
                  requestNodeSelection([element.data.fieldIdentifier.nodeId]);
                }}
              >
                <Icon as={CrosshairIcon} boxSize="3" />
              </IconButton>
              <FieldDescriptionAction element={element} projectGraph={projectGraph} />
              <IconButton
                aria-label="Toggle description"
                color={element.data.showDescription ? 'accent.solid' : undefined}
                size="2xs"
                title={element.data.showDescription ? 'Hide field description' : 'Show field description'}
                variant="ghost"
                onClick={() =>
                  dispatch({
                    action: {
                      elementId: element.id,
                      showDescription: !element.data.showDescription,
                      type: 'setNodeFieldShowDescription',
                    },
                    type: 'applyProjectGraphAction',
                  })
                }
              >
                <Icon as={InfoIcon} boxSize="3" />
              </IconButton>
            </>
          }
          index={index}
          isHovered={element.data.fieldIdentifier.nodeId === hoveredNodeId}
          isInvalid={invalidElementIds.has(element.id)}
          isSelected={selectedNodeIds.has(element.data.fieldIdentifier.nodeId)}
          parentId={parentId}
          title="Node Field"
        >
          <NodeFieldControl element={element} isLabelEditable projectGraph={projectGraph} />
        </BuilderCard>
      );
    }
    case 'heading': {
      return (
        <BuilderCard element={element} index={index} parentId={parentId} title="Heading">
          <Input
            aria-label="Form heading"
            fontSize="sm"
            fontWeight="700"
            placeholder="Heading"
            size="xs"
            value={element.data.content}
            variant="flushed"
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              dispatch({
                action: { content: event.currentTarget.value, elementId: element.id, type: 'setFormElementContent' },
                type: 'applyProjectGraphAction',
              })
            }
          />
        </BuilderCard>
      );
    }
    case 'text': {
      return (
        <BuilderCard element={element} index={index} parentId={parentId} title="Text">
          <Textarea
            aria-label="Form text"
            color="fg.muted"
            fontSize="2xs"
            minH="2.5rem"
            placeholder="Text"
            resize="vertical"
            size="xs"
            value={element.data.content}
            variant="flushed"
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
              dispatch({
                action: { content: event.currentTarget.value, elementId: element.id, type: 'setFormElementContent' },
                type: 'applyProjectGraphAction',
              })
            }
          />
        </BuilderCard>
      );
    }
    case 'divider': {
      return (
        <BuilderCard element={element} index={index} parentId={parentId} title="Divider">
          <Separator borderColor="border.subtle" />
        </BuilderCard>
      );
    }
  }
};

const AddElementMenu = () => {
  const dispatch = useWorkbenchDispatch();
  const add = (elementType: 'divider' | 'heading' | 'text' | 'container', layout?: 'row' | 'column') =>
    dispatch({ action: { elementType, layout, type: 'addFormElement' }, type: 'applyProjectGraphAction' });

  return (
    <Menu.Root positioning={{ placement: 'bottom-start' }}>
      <Menu.Trigger asChild>
        <Button size="2xs" variant="ghost">
          <Icon as={PlusIcon} boxSize="3" />
          Add form element
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <Menu.Content minW="11rem">
            <Menu.Item value="heading" onClick={() => add('heading')}>
              <Icon as={HeadingIcon} boxSize="3" />
              Heading
            </Menu.Item>
            <Menu.Item value="text" onClick={() => add('text')}>
              <Icon as={TextIcon} boxSize="3" />
              Text
            </Menu.Item>
            <Menu.Item value="divider" onClick={() => add('divider')}>
              <Icon as={MinusIcon} boxSize="3" />
              Divider
            </Menu.Item>
            <Menu.Item value="container-column" onClick={() => add('container', 'column')}>
              <Icon as={Columns2Icon} boxSize="3" />
              Container (column)
            </Menu.Item>
            <Menu.Item value="container-row" onClick={() => add('container', 'row')}>
              <Icon as={Rows2Icon} boxSize="3" />
              Container (row)
            </Menu.Item>
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const getInvalidNodeFieldElementIds = (
  projectGraph: ProjectGraphState,
  templatesSnapshot: ReturnType<typeof useInvocationTemplatesSnapshot>
): Set<string> => {
  const invalidElementIds = new Set<string>();

  if (templatesSnapshot.status !== 'loaded') {
    return invalidElementIds;
  }

  const connectedInputKeys = new Set(
    getResolvedWorkflowEdges(projectGraph.nodes, projectGraph.edges).map(
      (edge) => `${edge.target}:${edge.targetHandle}`
    )
  );

  for (const element of Object.values(projectGraph.form.elements)) {
    if (element.type !== 'node-field') {
      continue;
    }

    const { fieldName, nodeId } = element.data.fieldIdentifier;
    const node = projectGraph.nodes.find((candidate) => candidate.id === nodeId);

    if (!node || !isInvocationNode(node)) {
      invalidElementIds.add(element.id);
      continue;
    }

    const template = templatesSnapshot.templates[node.data.type]?.inputs[fieldName];

    if (!template) {
      invalidElementIds.add(element.id);
      continue;
    }

    const isConnected = connectedInputKeys.has(`${nodeId}:${fieldName}`);

    if (getWorkflowFieldInvalidReason({ isConnected, template, value: node.data.inputs[fieldName]?.value }) !== null) {
      invalidElementIds.add(element.id);
    }
  }

  return invalidElementIds;
};

export const FormBuilderTab = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const { hoveredNodeId, selectedNodeIds } = workflowSelectionStore.useSnapshot();
  const [draggingElementId, setDraggingElementId] = useState<string | null>(null);
  const dndContextValue = useMemo<BuilderDndContextValue>(
    () => ({ draggingElementId, setDraggingElementId }),
    [draggingElementId]
  );
  const selectedNodeIdSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);
  const invalidElementIds = useMemo(
    () => getInvalidNodeFieldElementIds(projectGraph, templatesSnapshot),
    [projectGraph, templatesSnapshot]
  );
  const rootChildren = getFormChildren(projectGraph.form);

  return (
    <BuilderDndContext value={dndContextValue}>
      <Stack gap="2" p="3" w="full">
        {rootChildren.length === 0 ? (
          <Text color="fg.subtle" fontSize="2xs">
            The form is empty. Pin fields from the Workflow editor's nodes, then arrange them here — drag card title
            bars to reorder, drop them into containers, and add headings or dividers below.
          </Text>
        ) : null}
        {rootChildren.map((element, index) => (
          <BuilderElement
            key={element.id}
            element={element}
            hoveredNodeId={hoveredNodeId}
            invalidElementIds={invalidElementIds}
            index={index}
            parentId={projectGraph.form.rootElementId}
            projectGraph={projectGraph}
            selectedNodeIds={selectedNodeIdSet}
          />
        ))}
        <AddElementMenu />
      </Stack>
    </BuilderDndContext>
  );
};
