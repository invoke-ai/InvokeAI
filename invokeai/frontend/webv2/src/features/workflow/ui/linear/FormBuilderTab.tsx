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
import { isSeedInputField } from '@features/workflow/graph';
import { useInvocationTemplatesSelector, type InvocationTemplatesSnapshot } from '@features/workflow/react';
import { requestNodeSelection, workflowSelectionStore } from '@features/workflow/ui/editor/selectionStore';
import { FieldDescriptionPopover } from '@features/workflow/ui/fields/FieldDescriptionPopover';
import { getWorkflowNodeChromeProps } from '@features/workflow/ui/nodeChrome';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { useWorkflowHostCommands } from '@features/workflow/ui/WorkflowUiContext';
import {
  getFormChildren,
  getResolvedWorkflowEdges,
  getWorkflowFieldInvalidReason,
  isShuffleableField,
} from '@features/workflow/utility';
import { Button, DropZone, IconButton, Tooltip } from '@platform/ui';
import { MenuContent } from '@platform/ui/Menu';
import {
  Columns2Icon,
  CrosshairIcon,
  DicesIcon,
  GripVerticalIcon,
  InfoIcon,
  PlusIcon,
  Rows2Icon,
  XIcon,
} from 'lucide-react';
import {
  createContext,
  memo,
  use,
  useCallback,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from 'react';
import { useTranslation } from 'react-i18next';

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
import {
  ADDABLE_FORM_ELEMENT_KEYS,
  FORM_ELEMENT_META,
  getFormElementTitleKey,
  type AddableFormElementKey,
} from './formElementMeta';
import { NodeFieldControl, useNodeFieldBinding } from './NodeFieldControl';

/**
 * Commit form moves only at DndContext onDragEnd so reparenting remounts cannot lose completion; all edits use the
 * document reducer.
 */

interface BuilderDndContextValue {
  activeElementId: string | null;
  form: WorkflowForm;
}

const BuilderDndContext = createContext<BuilderDndContextValue>({
  activeElementId: null,
  form: { elements: {}, rootElementId: '' },
});

/** Separate per-move drop targets from start/end drag state so only indicators rerender while the pointer moves. */
const BuilderDropTargetContext = createContext<FormDropTarget | null>(null);

/** The dragged card's `DragOverlay` ghost: a compact title bar following the pointer. */
const BuilderDragGhost = ({ element }: { element: WorkflowFormElement }) => {
  const { t } = useTranslation();

  return (
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
      <Text color="fg.muted" fontSize="xs" fontWeight="600" minW="0" truncate>
        {t(getFormElementTitleKey(element))}
      </Text>
    </HStack>
  );
};

/** Isolate edge indicators as the card's only per-move drop-target consumers. */
const CardDropEdgeIndicator = ({ elementId }: { elementId: string }) => {
  const dropTarget = use(BuilderDropTargetContext);
  const edge = dropTarget?.kind === 'edge' && dropTarget.elementId === elementId ? dropTarget.edge : null;

  if (!edge) {
    return null;
  }

  return (
    <Box
      bg="accent.solid"
      h="2px"
      left="0"
      pointerEvents="none"
      position="absolute"
      right="0"
      rounded="full"
      zIndex="1"
      {...(edge === 'above' ? { top: '-1px' } : { bottom: '-1px' })}
    />
  );
};

/** A builder card: typed title bar (drag handle + actions) over the element's content. */
const BuilderCardBase = ({
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
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const { activeElementId, form } = use(BuilderDndContext);
  const { attributes, listeners, setActivatorNodeRef, setNodeRef: setDragRef } = useDraggable({ id: element.id });
  // Set both draggable and activator refs on title bars so keyboard actions bubbling from child buttons cannot
  // lift cards.
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

  return (
    <Box ref={setDropRef} flex="1" minW="0" opacity={activeElementId === element.id ? 0.4 : 1} position="relative">
      <CardDropEdgeIndicator elementId={element.id} />
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
          <Text color="fg.muted" fontSize="xs" fontWeight="600" minW="0" truncate>
            {title}
          </Text>
          <Box flex="1" />
          <HStack flexShrink={0} gap="0" onPointerDown={(event) => event.stopPropagation()}>
            {extraActions}
            <Tooltip content={t('widgets.workflow.formBuilder.remove')}>
              <IconButton
                aria-label={t('widgets.workflow.formBuilder.remove')}
                size="sm"
                variant="ghost"
                onClick={() => editGraph({ elementId: element.id, type: 'removeFormElement' })}
              >
                <Icon as={XIcon} boxSize="3" />
              </IconButton>
            </Tooltip>
          </HStack>
        </HStack>
        <Box p="2" position="relative" zIndex="2">
          {children}
        </Box>
      </Box>
    </Box>
  );
};

/** Memoize cards so drop-target updates do not propagate beyond the indicator consumers. */
const BuilderCard = memo(BuilderCardBase);

/** Isolate container hover/copy updates from the rest of the drop zone. */
const ContainerDropZoneBody = ({
  canDrop,
  containerId,
  isEmpty,
  setNodeRef,
}: {
  canDrop: boolean;
  containerId: string;
  isEmpty: boolean;
  setNodeRef: (node: HTMLElement | null) => void;
}) => {
  const { t } = useTranslation();
  const dropTarget = use(BuilderDropTargetContext);
  const isActive = dropTarget?.kind === 'into' && dropTarget.containerId === containerId;

  return (
    <DropZone
      ref={setNodeRef}
      alignSelf="stretch"
      flex={isEmpty ? '1' : undefined}
      fontSize="xs"
      isOver={isActive}
      px="2"
      py="1.5"
      textAlign="center"
    >
      {canDrop ? t('widgets.workflow.formBuilder.dropHere') : t('widgets.workflow.formBuilder.containerEmpty')}
    </DropZone>
  );
};

/** Drop zone covering a container's body, appending at the end. Doubles as the empty-container hint. */
const ContainerDropZoneBase = ({ container, isEmpty }: { container: ContainerFormElement; isEmpty: boolean }) => {
  const { activeElementId, form } = use(BuilderDndContext);
  const canDrop = activeElementId !== null && !isFormDescendantOrSelf(form, activeElementId, container.id);
  const { setNodeRef } = useDroppable({
    disabled: activeElementId === null || isFormDescendantOrSelf(form, activeElementId, container.id),
    id: formIntoDroppableId(container.id),
  });

  if (!canDrop && !isEmpty) {
    return null;
  }

  return (
    <ContainerDropZoneBody canDrop={canDrop} containerId={container.id} isEmpty={isEmpty} setNodeRef={setNodeRef} />
  );
};

const ContainerDropZone = memo(ContainerDropZoneBase);

/** The shared description popover, bound through the form element. */
/** Seed fields own a dice already; other numeric fields opt into one per form element. */
const ShuffleToggleAction = ({
  element,
  projectGraph,
}: {
  element: NodeFieldFormElement;
  projectGraph: ProjectGraphState;
}) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const { template } = useNodeFieldBinding(element, projectGraph);

  if (!template || !isShuffleableField(template) || isSeedInputField(template)) {
    return null;
  }

  const { showShuffle } = element.data;

  return (
    <Tooltip
      content={
        showShuffle ? t('widgets.workflow.formBuilder.hideShuffle') : t('widgets.workflow.formBuilder.showShuffle')
      }
    >
      <IconButton
        aria-label={t('widgets.workflow.formBuilder.showShuffle')}
        aria-pressed={showShuffle}
        color={showShuffle ? 'accent.solid' : undefined}
        size="sm"
        variant="ghost"
        onClick={() => editGraph({ elementId: element.id, showShuffle: !showShuffle, type: 'setNodeFieldShowShuffle' })}
      >
        <Icon as={DicesIcon} boxSize="3" />
      </IconButton>
    </Tooltip>
  );
};

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

const BuilderElementBase = ({
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
  const { t } = useTranslation();
  const { widgets } = useWorkflowHostCommands();
  const { editGraph } = useProjectGraphCommands();
  const title = t(getFormElementTitleKey(element));

  switch (element.type) {
    case 'container': {
      const isRow = element.data.layout === 'row';
      const switchLayoutLabel = isRow
        ? t('widgets.workflow.formBuilder.switchToColumn')
        : t('widgets.workflow.formBuilder.switchToRow');

      return (
        <BuilderCard
          element={element}
          extraActions={
            <Tooltip content={switchLayoutLabel}>
              <IconButton
                aria-label={switchLayoutLabel}
                size="sm"
                variant="ghost"
                onClick={() =>
                  editGraph({ elementId: element.id, layout: isRow ? 'column' : 'row', type: 'setContainerLayout' })
                }
              >
                <Icon as={isRow ? Rows2Icon : Columns2Icon} boxSize="3" />
              </IconButton>
            </Tooltip>
          }
          title={title}
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
              <Tooltip content={t('widgets.workflow.formBuilder.zoomToNode')}>
                <IconButton
                  aria-label={t('widgets.workflow.formBuilder.zoomToNode')}
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    widgets.open({ region: 'center', widgetId: 'workflow' });
                    requestNodeSelection([element.data.fieldIdentifier.nodeId]);
                  }}
                >
                  <Icon as={CrosshairIcon} boxSize="3" />
                </IconButton>
              </Tooltip>
              <FieldDescriptionAction element={element} projectGraph={projectGraph} />
              <ShuffleToggleAction element={element} projectGraph={projectGraph} />
              <Tooltip
                content={
                  element.data.showDescription
                    ? t('widgets.workflow.formBuilder.hideDescription')
                    : t('widgets.workflow.formBuilder.showDescription')
                }
              >
                <IconButton
                  aria-label={t('widgets.workflow.formBuilder.showDescription')}
                  aria-pressed={element.data.showDescription}
                  color={element.data.showDescription ? 'accent.solid' : undefined}
                  size="sm"
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
              </Tooltip>
            </>
          }
          isHovered={element.data.fieldIdentifier.nodeId === hoveredNodeId}
          isInvalid={invalidElementIds.has(element.id)}
          isSelected={selectedNodeIds.has(element.data.fieldIdentifier.nodeId)}
          title={title}
        >
          <NodeFieldControl element={element} isLabelEditable projectGraph={projectGraph} />
        </BuilderCard>
      );
    }
    case 'heading': {
      return (
        <BuilderCard element={element} title={title}>
          <Input
            aria-label={t('widgets.workflow.formBuilder.headingLabel')}
            fontSize="lg"
            fontWeight="700"
            placeholder={t('widgets.workflow.formBuilder.headingPlaceholder')}
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
        <BuilderCard element={element} title={title}>
          <Textarea
            aria-label={t('widgets.workflow.formBuilder.textLabel')}
            color="fg.muted"
            fontSize="xs"
            minH="2.5rem"
            placeholder={t('widgets.workflow.formBuilder.textPlaceholder')}
            resize="vertical"
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
        <BuilderCard element={element} title={title}>
          <Separator borderColor="border.subtle" />
        </BuilderCard>
      );
    }
  }
};

/**
 * Memoized elements retain stable mid-drag props; only direct drop-target context consumers rerender on pointer
 * movement.
 */
const BuilderElement = memo(BuilderElementBase);

const AddElementMenu = () => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const add = (key: AddableFormElementKey) =>
    editGraph(
      key === 'container-column' || key === 'container-row'
        ? { elementType: 'container', layout: key === 'container-row' ? 'row' : 'column', type: 'addFormElement' }
        : { elementType: key, type: 'addFormElement' }
    );

  return (
    <Menu.Root positioning={{ placement: 'bottom-start' }}>
      <Menu.Trigger asChild>
        <Button size="sm" variant="ghost">
          <Icon as={PlusIcon} boxSize="3" />
          {t('widgets.workflow.formBuilder.addElement')}
        </Button>
      </Menu.Trigger>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="11rem">
            {ADDABLE_FORM_ELEMENT_KEYS.map((key) => (
              <Menu.Item key={key} value={key} onClick={() => add(key)}>
                <Icon as={FORM_ELEMENT_META[key].icon} boxSize="3" />
                {t(FORM_ELEMENT_META[key].labelKey)}
              </Menu.Item>
            ))}
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

export const getInvalidNodeFieldElementIds = (
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

    const template = node.data.dynamicInputTemplates?.[fieldName] ?? templates[node.data.type]?.inputs[fieldName];

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
  const { t } = useTranslation();
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
  // Capture collision pointerCoordinates; drag delta includes scroll adjustment and cannot be added to clientY
  // without overshooting.
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

    // Use true pointer coordinates when available; keyboard drags use the translated card center.
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
    () => ({ activeElementId, form: projectGraph.form }),
    [activeElementId, projectGraph.form]
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
        <BuilderDropTargetContext value={dropTarget}>
          <Stack gap="2" p="3" w="full">
            {rootChildren.length === 0 ? (
              <Text color="fg.muted" fontSize="xs">
                {t('widgets.workflow.formBuilder.empty')}
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
        </BuilderDropTargetContext>
      </BuilderDndContext>
      <DragOverlay dropAnimation={null}>
        {activeElement ? <BuilderDragGhost element={activeElement} /> : null}
      </DragOverlay>
    </DndContext>
  );
};
