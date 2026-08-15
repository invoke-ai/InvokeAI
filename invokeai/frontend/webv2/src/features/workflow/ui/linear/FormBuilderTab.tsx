import { Box, HStack, Icon, Input, Menu, Portal, Separator, Stack, Text, Textarea } from '@chakra-ui/react';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import {
  DndContext,
  DragOverlay,
  KeyboardSensor,
  MeasuringStrategy,
  PointerSensor,
  pointerWithin,
  rectIntersection,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type CollisionDetection,
  type DragEndEvent,
  type DragMoveEvent,
  type DragStartEvent,
} from '@dnd-kit/core';
import {
  isInvocationNode,
  type ContainerFormElement,
  type InvocationTemplates,
  type NodeFieldFormElement,
  type ProjectGraphState,
  type WorkflowForm,
  type WorkflowFormElement,
} from '@features/workflow/contracts';
import { useInvocationTemplatesSelector, type InvocationTemplatesSnapshot } from '@features/workflow/react';
import { getWorkflowNodeChromeProps } from '@features/workflow/ui/editor/nodeChrome';
import { requestNodeSelection, workflowSelectionStore } from '@features/workflow/ui/editor/selectionStore';
import { FieldDescriptionPopover } from '@features/workflow/ui/fields/FieldDescriptionPopover';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowHostCommands } from '@features/workflow/ui/WorkflowUiContext';
import { getFormChildren, getResolvedWorkflowEdges, getWorkflowFieldInvalidReason } from '@features/workflow/utility';
import { Button, DropZone, IconButton } from '@platform/ui';
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
import { createContext, use, useCallback, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from 'react';

import {
  formEdgeDroppableId,
  formIntoDroppableId,
  getFormDropEdge,
  isFormDescendantOrSelf,
  parseFormDroppableId,
  pickInnermostFormCollision,
  resolveFormDrop,
  type FormDropTarget,
} from './formBuilderDnd';
import { NodeFieldControl, useNodeFieldBinding } from './NodeFieldControl';

/**
 * The form builder: edit mode of the Linear UI. Every element renders as a
 * card with its own title bar — type label on the left, actions on the right,
 * content below — mirroring the legacy builder. Cards reorder and reparent by
 * dragging their title bar (drop indicators above/below, containers accept
 * drops into their body) via dnd-kit: moves happen only at `onDragEnd`, which
 * fires at the `DndContext` level regardless of whether the drop reparents
 * (and therefore remounts) the dragged card. All edits go through the
 * project graph document reducer.
 */

interface BuilderDndContextValue {
  activeElementId: string | null;
  dropTarget: FormDropTarget | null;
  form: WorkflowForm;
}

const BuilderDndContext = createContext<BuilderDndContextValue>({
  activeElementId: null,
  dropTarget: null,
  form: { elements: {}, rootElementId: '' },
});

/** Title shown in a card's title bar and the drag ghost. Shared so the two never drift. */
const getFormElementTitle = (element: WorkflowFormElement): string => {
  switch (element.type) {
    case 'container':
      return `Container (${element.data.layout} layout)`;
    case 'node-field':
      return 'Node Field';
    case 'heading':
      return 'Heading';
    case 'text':
      return 'Text';
    case 'divider':
      return 'Divider';
  }
};

/** The dragged card's `DragOverlay` ghost: a compact title bar following the pointer. */
const BuilderDragGhost = ({ element }: { element: WorkflowFormElement }) => (
  <HStack
    bg="bg.muted"
    borderColor="border.subtle"
    borderWidth="1px"
    cursor="grabbing"
    gap="1"
    opacity={0.85}
    px="1.5"
    py="0.5"
    rounded="md"
    shadow="md"
  >
    <Icon as={GripVerticalIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
    <Text color="fg.muted" fontSize="2xs" fontWeight="600" minW="0" truncate>
      {getFormElementTitle(element)}
    </Text>
  </HStack>
);

/** A builder card: typed title bar (drag handle + actions) over the element's content. */
const BuilderCard = ({
  children,
  element,
  extraActions,
  isHovered,
  isInvalid,
  isSelected,
  title,
}: {
  children: ReactNode;
  element: WorkflowFormElement;
  extraActions?: ReactNode;
  isHovered?: boolean;
  isInvalid?: boolean;
  isSelected?: boolean;
  title: string;
}) => {
  const { editGraph } = useProjectGraphCommands();
  const { activeElementId, dropTarget, form } = use(BuilderDndContext);
  const { attributes, listeners, setActivatorNodeRef, setNodeRef: setDragRef } = useDraggable({ id: element.id });
  // The title bar is both the draggable node and its own drag handle, so it
  // needs both refs: `setActivatorNodeRef` is what makes `KeyboardSensor`
  // enforce `event.target === activator` (dnd-kit's `KeyboardSensor.activators`
  // check) — without it, `Space`/`Enter` bubbling up from the action buttons
  // this title bar contains (Remove, Zoom to node, etc.) would also lift the
  // card, the keyboard equivalent of the `onPointerDown` `stopPropagation`
  // guard those buttons already carry for pointer drags.
  const setDragHandleRef = useCallback(
    (node: HTMLElement | null) => {
      setDragRef(node);
      setActivatorNodeRef(node);
    },
    [setActivatorNodeRef, setDragRef]
  );
  const { setNodeRef: setDropRef } = useDroppable({
    disabled: activeElementId !== null && isFormDescendantOrSelf(form, activeElementId, element.id),
    id: formEdgeDroppableId(element.id),
  });
  const edgeForThisCard = dropTarget?.kind === 'edge' && dropTarget.elementId === element.id ? dropTarget.edge : null;

  return (
    <Box ref={setDropRef} flex="1" minW="0" opacity={activeElementId === element.id ? 0.4 : 1} position="relative">
      {edgeForThisCard ? (
        <Box
          bg="accent.solid"
          h="2px"
          left="0"
          pointerEvents="none"
          position="absolute"
          right="0"
          rounded="full"
          zIndex="1"
          {...(edgeForThisCard === 'above' ? { top: '-1px' } : { bottom: '-1px' })}
        />
      ) : null}
      <Box
        overflow="hidden"
        position="relative"
        rounded="md"
        {...getWorkflowNodeChromeProps({ invalid: Boolean(isInvalid), selected: Boolean(isHovered || isSelected) })}
      >
        <HStack
          ref={setDragHandleRef}
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
          {...attributes}
          {...listeners}
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
              onClick={() => editGraph({ elementId: element.id, type: 'removeFormElement' })}
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
  const { activeElementId, dropTarget, form } = use(BuilderDndContext);
  const canDrop = activeElementId !== null && !isFormDescendantOrSelf(form, activeElementId, container.id);
  const { setNodeRef } = useDroppable({
    disabled: activeElementId === null || isFormDescendantOrSelf(form, activeElementId, container.id),
    id: formIntoDroppableId(container.id),
  });
  const isActive = dropTarget?.kind === 'into' && dropTarget.containerId === container.id;

  if (!canDrop && !isEmpty) {
    return null;
  }

  return (
    <DropZone
      ref={setNodeRef}
      alignSelf="stretch"
      flex={isEmpty ? '1' : undefined}
      fontSize="2xs"
      isOver={isActive}
      px="2"
      py="1.5"
      textAlign="center"
    >
      {canDrop ? 'Drop here' : 'Empty container — drag elements here'}
    </DropZone>
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
  hoveredNodeId,
  invalidElementIds,
  projectGraph,
  selectedNodeIds,
}: {
  element: WorkflowFormElement;
  hoveredNodeId: string | null;
  invalidElementIds: Set<string>;
  projectGraph: ProjectGraphState;
  selectedNodeIds: Set<string>;
}) => {
  const { widgets } = useWorkflowHostCommands();
  const { editGraph } = useProjectGraphCommands();

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
                editGraph({ elementId: element.id, layout: isRow ? 'column' : 'row', type: 'setContainerLayout' })
              }
            >
              <Icon as={isRow ? Rows2Icon : Columns2Icon} boxSize="3" />
            </IconButton>
          }
          title={getFormElementTitle(element)}
        >
          <Stack align={isRow ? 'stretch' : undefined} direction={isRow ? 'row' : 'column'} gap="2" w="full">
            {getFormChildren(projectGraph.form, element.id).map((child) => (
              <BuilderElement
                key={child.id}
                element={child}
                hoveredNodeId={hoveredNodeId}
                invalidElementIds={invalidElementIds}
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
                  widgets.open({ region: 'center', widgetId: 'workflow' });
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
                  editGraph({
                    elementId: element.id,
                    showDescription: !element.data.showDescription,
                    type: 'setNodeFieldShowDescription',
                  })
                }
              >
                <Icon as={InfoIcon} boxSize="3" />
              </IconButton>
            </>
          }
          isHovered={element.data.fieldIdentifier.nodeId === hoveredNodeId}
          isInvalid={invalidElementIds.has(element.id)}
          isSelected={selectedNodeIds.has(element.data.fieldIdentifier.nodeId)}
          title={getFormElementTitle(element)}
        >
          <NodeFieldControl element={element} isLabelEditable projectGraph={projectGraph} />
        </BuilderCard>
      );
    }
    case 'heading': {
      return (
        <BuilderCard element={element} title={getFormElementTitle(element)}>
          <Input
            aria-label="Form heading"
            fontSize="sm"
            fontWeight="700"
            placeholder="Heading"
            size="xs"
            value={element.data.content}
            variant="flushed"
            onChange={(event: ChangeEvent<HTMLInputElement>) =>
              editGraph({ content: event.currentTarget.value, elementId: element.id, type: 'setFormElementContent' })
            }
          />
        </BuilderCard>
      );
    }
    case 'text': {
      return (
        <BuilderCard element={element} title={getFormElementTitle(element)}>
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
              editGraph({ content: event.currentTarget.value, elementId: element.id, type: 'setFormElementContent' })
            }
          />
        </BuilderCard>
      );
    }
    case 'divider': {
      return (
        <BuilderCard element={element} title={getFormElementTitle(element)}>
          <Separator borderColor="border.subtle" />
        </BuilderCard>
      );
    }
  }
};

const AddElementMenu = () => {
  const { editGraph } = useProjectGraphCommands();
  const add = (elementType: 'divider' | 'heading' | 'text' | 'container', layout?: 'row' | 'column') =>
    editGraph({ elementType, layout, type: 'addFormElement' });

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
  templatesStatus: InvocationTemplatesSnapshot['status'],
  templates: InvocationTemplates
): Set<string> => {
  const invalidElementIds = new Set<string>();

  if (templatesStatus !== 'loaded') {
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

    const template = templates[node.data.type]?.inputs[fieldName];

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
  const templatesStatus = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const hoveredNodeId = workflowSelectionStore.useSelector((snapshot) => snapshot.hoveredNodeId);
  const selectedNodeIds = workflowSelectionStore.useSelector((snapshot) => snapshot.selectedNodeIds);
  const { editGraph } = useProjectGraphCommands();
  const [activeElementId, setActiveElementId] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<FormDropTarget | null>(null);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor)
  );
  // Droppable rects freeze at drag start by default; the builder's drop
  // indicators and container hints appear mid-drag, so re-measure continuously.
  const measuring = useMemo(() => ({ droppable: { strategy: MeasuringStrategy.Always } }), []);
  const form = projectGraph.form;
  // `DragMoveEvent.delta` is scroll-adjusted (translate + the panel's scroll
  // offset baked in), so `activatorEvent.clientY + delta.y` overshoots the
  // real pointer once the panel auto-scrolls while `over.rect` stays in
  // fresh viewport coordinates. `collisionDetection`'s `args.pointerCoordinates`
  // is the one place dnd-kit hands back the true (already scroll-correct)
  // pointer position, so it's captured here and read in `handleDragMove`.
  const pointerYRef = useRef<number | null>(null);
  const collisionDetection: CollisionDetection = useCallback(
    (args) => {
      pointerYRef.current = args.pointerCoordinates?.y ?? null;

      const within = pointerWithin(args);
      const candidates = within.length > 0 ? within : rectIntersection(args);
      const picked = pickInnermostFormCollision(
        candidates.map((collision) => ({ id: String(collision.id) })),
        form
      );

      return picked === null ? candidates : candidates.filter((collision) => String(collision.id) === picked);
    },
    [form]
  );

  const handleDragStart = useCallback((event: DragStartEvent) => {
    setActiveElementId(String(event.active.id));
  }, []);
  const handleDragMove = useCallback((event: DragMoveEvent) => {
    const { active, over } = event;

    if (!over) {
      setDropTarget(null);
      return;
    }

    const parsed = parseFormDroppableId(String(over.id));

    if (!parsed) {
      setDropTarget(null);
      return;
    }

    if (parsed.kind === 'into') {
      setDropTarget({ containerId: parsed.containerId, kind: 'into' });
      return;
    }

    // The true (scroll-correct) pointer position when the drag has one
    // (`PointerSensor`, captured from `collisionDetection`'s
    // `pointerCoordinates`); a `KeyboardSensor` drag has no pointer, so fall
    // back to the dragged card's translated center.
    const activeRect = active.rect.current.translated;
    const referenceY = pointerYRef.current ?? (activeRect ? activeRect.top + activeRect.height / 2 : over.rect.top);

    setDropTarget({
      edge: getFormDropEdge(referenceY, over.rect.top, over.rect.height),
      elementId: parsed.elementId,
      kind: 'edge',
    });
  }, []);
  const clearDrag = useCallback(() => {
    setActiveElementId(null);
    setDropTarget(null);
  }, []);
  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const activeId = String(event.active.id);

      if (dropTarget) {
        const resolved = resolveFormDrop(projectGraph.form, activeId, dropTarget);

        if (resolved) {
          editGraph({
            elementId: activeId,
            index: resolved.index,
            parentId: resolved.parentId,
            type: 'moveFormElementTo',
          });
        }
      }

      clearDrag();
    },
    [clearDrag, dropTarget, editGraph, projectGraph.form]
  );

  const dndContextValue = useMemo<BuilderDndContextValue>(
    () => ({ activeElementId, dropTarget, form: projectGraph.form }),
    [activeElementId, dropTarget, projectGraph.form]
  );
  const selectedNodeIdSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);
  const invalidElementIds = useMemo(
    () => getInvalidNodeFieldElementIds(projectGraph, templatesStatus, templates),
    [projectGraph, templatesStatus, templates]
  );
  const rootChildren = getFormChildren(projectGraph.form);
  const activeElement = activeElementId ? projectGraph.form.elements[activeElementId] : undefined;

  return (
    <DndContext
      collisionDetection={collisionDetection}
      measuring={measuring}
      sensors={sensors}
      onDragCancel={clearDrag}
      onDragEnd={handleDragEnd}
      onDragMove={handleDragMove}
      onDragStart={handleDragStart}
    >
      <BuilderDndContext value={dndContextValue}>
        <Stack gap="2" p="3" w="full">
          {rootChildren.length === 0 ? (
            <Text color="fg.subtle" fontSize="2xs">
              The form is empty. Pin fields from the Workflow editor's nodes, then arrange them here — drag card title
              bars to reorder, drop them into containers, and add headings or dividers below.
            </Text>
          ) : null}
          {rootChildren.map((element) => (
            <BuilderElement
              key={element.id}
              element={element}
              hoveredNodeId={hoveredNodeId}
              invalidElementIds={invalidElementIds}
              projectGraph={projectGraph}
              selectedNodeIds={selectedNodeIdSet}
            />
          ))}
          <AddElementMenu />
        </Stack>
      </BuilderDndContext>
      <DragOverlay dropAnimation={null}>
        {activeElement ? <BuilderDragGhost element={activeElement} /> : null}
      </DragOverlay>
    </DndContext>
  );
};
