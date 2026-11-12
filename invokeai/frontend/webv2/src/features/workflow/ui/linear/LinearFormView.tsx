import type { CollapsibleFormElement, ProjectGraphState, WorkflowFormElement } from '@features/workflow/contracts';

import { Collapsible, Separator, Stack, Text } from '@chakra-ui/react';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import {
  useWorkflowHostCommands,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import { getFormChildren } from '@features/workflow/utility';
import { Button } from '@platform/ui';
import { ChevronRightIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getFormSectionOpenState, updateFormSectionOpenState } from './formSections';
import { getInvalidFormFieldIds, sectionHasInvalidField } from './formSectionStatus';
import { LinkifiedText } from './LinkifiedText';
import { NodeFieldControl } from './NodeFieldControl';

type FormViewContext = {
  projectGraph: ProjectGraphState;
  columnGap: '1' | '3';
  openSections: Readonly<Record<string, boolean>>;
  invalidFieldIds: ReadonlySet<string>;
  onSectionOpenChange: (sectionId: string, open: boolean) => void;
};

type FormLayout = 'row' | 'column';

const OPEN_INDICATOR_STYLES = { transform: 'rotate(90deg)' };
const SECTION_OPEN_STYLES = { bg: 'bg.muted', rounded: 'sm' };
const SECTION_SEPARATOR_STYLES = {
  '&::before': {
    content: '""',
    position: 'absolute',
    insetInline: 0,
    top: 'calc(var(--workflow-form-section-gap) / -2)',
    borderTopWidth: '1px',
    pointerEvents: 'none',
    opacity: 0,
    transition: 'opacity var(--wb-motion-duration-slow) ease',
  },
  '&[data-section-boundary]::before': { opacity: 1 },
} as const;

export const LinearFormView = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const { t } = useTranslation();
  const { widgets } = useWorkflowHostCommands();
  const { project } = useWorkflowUi();
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const projectId = useWorkflowProjectSelector((snapshot) => snapshot.id);
  const storedOpenSections = useWorkflowProjectSelector((snapshot) => snapshot.workflowValues?.formSectionsOpen);
  const rootChildren = getFormChildren(projectGraph.form);
  const root = projectGraph.form.elements[projectGraph.form.rootElementId];
  const layout = root?.type === 'container' ? root.data.layout : 'column';
  const hasCollapsibleSections = useMemo(
    () => Object.values(projectGraph.form.elements).some((element) => element.type === 'collapsible'),
    [projectGraph.form.elements]
  );
  const columnGap: '1' | '3' = hasCollapsibleSections ? '1' : '3';
  const invalidFieldIds = useMemo(() => getInvalidFormFieldIds(projectGraph, templates), [projectGraph, templates]);
  const openSections = useMemo(
    () => getFormSectionOpenState({ formSectionsOpen: storedOpenSections }, projectGraph.id),
    [projectGraph.id, storedOpenSections]
  );
  const onOpenWorkflowEditorClick = useCallback(
    () => widgets.open({ region: 'center', widgetId: 'workflow' }),
    [widgets]
  );
  const onSectionOpenChange = useCallback(
    (sectionId: string, open: boolean) => {
      const snapshot = project.getSnapshot();
      if (snapshot.id !== projectId || snapshot.projectGraph?.id !== projectGraph.id) {
        return;
      }
      widgets.patchValues(
        'workflow',
        updateFormSectionOpenState(snapshot.workflowValues, projectGraph.id, sectionId, open)
      );
    },
    [project, projectGraph.id, projectId, widgets]
  );
  const context = useMemo(
    () => ({ projectGraph, columnGap, invalidFieldIds, openSections, onSectionOpenChange }),
    [projectGraph, columnGap, invalidFieldIds, openSections, onSectionOpenChange]
  );

  if (rootChildren.length === 0) {
    return (
      <Stack gap="2" p="3">
        <Text color="fg.muted" fontSize="xs">
          {t('widgets.workflow.formView.empty')}
        </Text>
        <Button size="sm" variant="outline" w="fit-content" onClick={onOpenWorkflowEditorClick}>
          {t('widgets.workflow.formView.openEditor')}
        </Button>
      </Stack>
    );
  }

  return (
    <Stack direction={layout} gap={layout === 'row' ? '3' : columnGap} px="1" py="3" minW="0">
      <ViewChildren context={context} elements={rootChildren} layout={layout} />
    </Stack>
  );
};

const ViewChildren = ({
  context,
  elements,
  layout,
  gap = `var(--chakra-spacing-${context.columnGap})`,
}: {
  context: FormViewContext;
  elements: readonly WorkflowFormElement[];
  layout: FormLayout;
  gap?: string;
}) =>
  elements.map((element, index) => {
    const showBoundary =
      layout === 'column' &&
      isClosedSectionEdge(elements[index - 1], 'end', context) &&
      isClosedSectionEdge(element, 'start', context);
    return (
      <ViewElement
        key={element.id}
        context={context}
        element={element}
        layout={layout}
        showBoundary={showBoundary}
        boundaryGap={gap}
      />
    );
  });

const isClosedSectionEdge = (
  element: WorkflowFormElement | undefined,
  edge: 'start' | 'end',
  context: FormViewContext
): boolean => {
  if (!element) {
    return false;
  }
  if (element.type === 'collapsible') {
    return context.openSections[element.id] === false;
  }
  if (element.type !== 'container') {
    return false;
  }
  // Layout-only wrappers do not break visual adjacency; multi-item rows have no single vertical edge.
  const children = getFormChildren(context.projectGraph.form, element.id);
  if (element.data.layout === 'row' && children.length !== 1) {
    return false;
  }
  return isClosedSectionEdge(edge === 'start' ? children[0] : children.at(-1), edge, context);
};

const FormSection = ({
  context,
  element,
  layout,
  showBoundary,
  boundaryGap,
}: {
  context: FormViewContext;
  element: CollapsibleFormElement;
  layout: FormLayout;
  showBoundary: boolean;
  boundaryGap: string;
}) => {
  const { t } = useTranslation();
  const children = getFormChildren(context.projectGraph.form, element.id);
  const open = context.openSections[element.id] ?? true;
  const hasInvalidField = sectionHasInvalidField(children, context.projectGraph, context.invalidFieldIds);
  const boundaryStyles = useMemo(
    () => ({ ...SECTION_SEPARATOR_STYLES, '--workflow-form-section-gap': boundaryGap }),
    [boundaryGap]
  );
  const onOpenChange = useCallback(
    ({ open: nextOpen }: { open: boolean }) => context.onSectionOpenChange(element.id, nextOpen),
    [context, element.id]
  );

  return (
    <Collapsible.Root
      css={layout === 'column' ? boundaryStyles : undefined}
      data-section-boundary={showBoundary ? '' : undefined}
      _open={SECTION_OPEN_STYLES}
      flex={layout === 'row' ? '1' : undefined}
      minW="0"
      open={open}
      position="relative"
      transition="background var(--wb-motion-duration-slow) ease, border-radius var(--wb-motion-duration-slow) ease"
      onOpenChange={onOpenChange}
    >
      <Collapsible.Trigger alignItems="center" display="flex" gap="2" h="8" px="1.5" textAlign="start" w="full">
        <Collapsible.Indicator _open={OPEN_INDICATOR_STYLES} transition="transform var(--wb-motion-duration-slow)">
          <ChevronRightIcon size="14" />
        </Collapsible.Indicator>
        <Text
          as="span"
          color="fg.muted"
          flex="1"
          fontSize="xs"
          fontWeight="bold"
          letterSpacing="widest"
          lineHeight="1"
          minW="0"
          textTransform="uppercase"
          truncate
        >
          {element.data.content || t('widgets.workflow.formView.unnamedSection', { defaultValue: 'Section' })}
        </Text>
        {!open && hasInvalidField ? (
          <Text as="span" color="fg.error" fontSize="xs" flexShrink="0">
            {t('widgets.workflow.formView.needsAttention', { defaultValue: 'Needs attention' })}
          </Text>
        ) : null}
      </Collapsible.Trigger>
      <Collapsible.Content>
        <Stack borderColor="bg.subtle" borderTopWidth="1px" direction="column" gap="3" p="2" minW="0">
          <ViewChildren context={context} elements={children} layout="column" gap="var(--chakra-spacing-3)" />
        </Stack>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

const ViewElement = ({
  element,
  context,
  layout,
  showBoundary,
  boundaryGap,
}: {
  element: WorkflowFormElement;
  context: FormViewContext;
  layout: FormLayout;
  showBoundary: boolean;
  boundaryGap: string;
}) => {
  const boundaryStyles = useMemo(
    () => ({ ...SECTION_SEPARATOR_STYLES, '--workflow-form-section-gap': boundaryGap }),
    [boundaryGap]
  );
  switch (element.type) {
    case 'collapsible':
      return (
        <FormSection
          context={context}
          element={element}
          layout={layout}
          showBoundary={showBoundary}
          boundaryGap={boundaryGap}
        />
      );
    case 'container':
      return (
        <Stack
          css={boundaryStyles}
          data-section-boundary={showBoundary ? '' : undefined}
          direction={element.data.layout}
          gap={element.data.layout === 'row' ? '3' : context.columnGap}
          minW="0"
          position="relative"
        >
          <ViewChildren
            context={context}
            elements={getFormChildren(context.projectGraph.form, element.id)}
            layout={element.data.layout}
          />
        </Stack>
      );
    case 'node-field':
      return <NodeFieldControl element={element} projectGraph={context.projectGraph} />;
    case 'heading':
      return (
        <Text fontSize="lg" fontWeight="700">
          {element.data.content}
        </Text>
      );
    case 'text':
      return (
        <LinkifiedText color="fg.muted" fontSize="xs" whiteSpace="pre-wrap">
          {element.data.content}
        </LinkifiedText>
      );
    case 'divider':
      return <Separator borderColor="border.subtle" />;
  }
};
