import type {
  CollapsibleFormElement,
  ProjectGraphState,
  WorkflowForm,
  WorkflowFormElement,
} from '@features/workflow/contracts';

import { Box, Collapsible, Flex, Icon, Separator, Stack, Text } from '@chakra-ui/react';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import {
  useWorkflowHostCommands,
  useWorkflowProjectSelector,
  useWorkflowUi,
} from '@features/workflow/ui/WorkflowUiContext';
import { getFormChildren } from '@features/workflow/utility';
import { Button } from '@platform/ui';
import { ChevronRightIcon, CircleAlertIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getFormSectionOpenState, updateFormSectionOpenState } from './formSections';
import { getInvalidFormFieldIds, sectionHasInvalidField } from './formSectionStatus';
import { LinkifiedText } from './LinkifiedText';
import { NodeFieldControl } from './NodeFieldControl';

type FormViewContext = {
  projectGraph: ProjectGraphState;
  openSections: Readonly<Record<string, boolean>>;
  invalidFieldIds: ReadonlySet<string>;
  onSectionOpenChange: (sectionId: string, open: boolean) => void;
};

type FormLayout = 'row' | 'column';

// Match Generate and Video collapsible trays (hairlines between folded rows) while keeping 10 px standalone spacing.
const ROOT_INSET = '1';
const ROOT_HEADING_STYLES = {
  '&:not(:first-child)': { borderTopWidth: '1px', mx: `-${ROOT_INSET}`, pt: '3', px: ROOT_INSET },
  // An authored divider or a closed collapsible section above already drew the line. With equal specificity, this must
  // follow the rule above.
  '[data-form-divider] + &, .workflow-form-section[data-state="closed"] + &': {
    borderTopWidth: 0,
    mx: 0,
    pt: 0,
    px: 0,
  },
} as const;

const CHIP_SHOWN_BG = 'color-mix(in oklab, {colors.bg} 50%, {colors.bg.subtle})';
const SEAM_RADIUS = '2px';
const TRAY_BG = 'bg.muted';
const OPEN_INDICATOR_STYLES = { transform: 'rotate(90deg)' };
const POINTED = ':has(> [data-part="trigger"]:hover)';
const SECTION_STYLES = {
  [`&:is([data-state="open"], ${POINTED}), .workflow-form-section:is([data-state="open"], ${POINTED}) + &, &[data-prev-open]`]:
    {
      borderTopColor: 'transparent',
    },
  [`&:is([data-state="open"], ${POINTED})`]: { borderBottomColor: 'transparent' },
} as const;
const TRIGGER_STYLES = {
  '&[data-state="open"]': { color: 'fg', pb: '2px' },
  '&[data-state="open"] > [data-section-chip]': { bg: CHIP_SHOWN_BG, borderBottomRadius: SEAM_RADIUS },
  '&:hover > [data-section-chip]': { bg: TRAY_BG },
  '&:not([data-state="open"]):hover': { color: 'fg' },
  '& > [data-section-chip][data-issues]': { bg: 'bg.error', color: 'fg.error' },
  '&:hover > [data-section-chip][data-issues]': { bg: 'red.hoverTint/20' },
  '&:focus-visible': { outline: 'none' },
  '&:focus-visible > [data-section-chip]': {
    outline: '2px solid',
    outlineColor: 'accent.solid',
    outlineOffset: '-2px',
  },
} as const;
const FIELDS_STYLES = {
  '--wb-field-surface': '{colors.bg.muted}',
  transformOrigin: 'top',
  '[data-part="content"][data-state="open"] > &': {
    animationDuration: 'moderate',
    animationName: 'wb-section-scale-in',
  },
  '[data-part="content"][data-state="closed"] > &': {
    animationDuration: 'moderate',
    animationName: 'wb-section-scale-out',
  },
} as const;

export const LinearFormView = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const { t } = useTranslation();
  const { widgets } = useWorkflowHostCommands();
  const { project } = useWorkflowUi();
  const templatesStatus = useInvocationTemplatesSelector((snapshot) => snapshot.status);
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const projectId = useWorkflowProjectSelector((snapshot) => snapshot.id);
  const storedOpenSections = useWorkflowProjectSelector((snapshot) => snapshot.workflowValues?.formSectionsOpen);
  const rootChildren = getFormChildren(projectGraph.form);
  const root = projectGraph.form.elements[projectGraph.form.rootElementId];
  const layout = root?.type === 'container' ? root.data.layout : 'column';
  const invalidFieldIds = useMemo(
    () => getInvalidFormFieldIds(projectGraph, templates, templatesStatus),
    [projectGraph, templates, templatesStatus]
  );
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
      const validSectionIds = new Set(
        Object.values(snapshot.projectGraph.form.elements)
          .filter((element) => element.type === 'collapsible')
          .map((element) => element.id)
      );
      widgets.patchValues(
        'workflow',
        updateFormSectionOpenState(snapshot.workflowValues, projectGraph.id, sectionId, open, validSectionIds)
      );
    },
    [project, projectGraph.id, projectId, widgets]
  );
  const context = useMemo(
    () => ({ projectGraph, invalidFieldIds, openSections, onSectionOpenChange }),
    [projectGraph, invalidFieldIds, openSections, onSectionOpenChange]
  );

  if (rootChildren.length === 0) {
    return (
      // Same inset and color as the Edit tab's empty state and as this view once it has fields.
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
    <Stack direction={layout} gap="2.5" px={ROOT_INSET} py="3" minW="0">
      <ViewChildren context={context} elements={rootChildren} isRootGroup layout={layout} />
    </Stack>
  );
};

const ViewChildren = ({
  context,
  elements,
  hasNextSection = false,
  isRootGroup = false,
  layout,
  prevSectionOpen = false,
}: {
  context: FormViewContext;
  elements: readonly WorkflowFormElement[];
  hasNextSection?: boolean;
  isRootGroup?: boolean;
  layout: FormLayout;
  prevSectionOpen?: boolean;
}) =>
  elements.map((element, index) => {
    const previousSection =
      layout === 'column' ? getEdgeSection(elements[index - 1], 'end', context.projectGraph.form) : null;
    const currentSection = layout === 'column' ? getEdgeSection(element, 'start', context.projectGraph.form) : null;
    const nextSection =
      layout === 'column' ? getEdgeSection(elements[index + 1], 'start', context.projectGraph.form) : null;
    const isAdjacentSection = previousSection !== null && currentSection !== null;
    const isPreviousSectionOpen =
      previousSection !== null ? (context.openSections[previousSection.id] ?? true) : index === 0 && prevSectionOpen;
    const continuesToSection = nextSection !== null || (index === elements.length - 1 && hasNextSection);

    return (
      <ViewElement
        key={element.id}
        context={context}
        continuesToSection={continuesToSection}
        element={element}
        isAdjacentSection={isAdjacentSection}
        isPreviousSectionOpen={isPreviousSectionOpen}
        isRoot={isRootGroup}
        layout={layout}
      />
    );
  });

const getEdgeSection = (
  element: WorkflowFormElement | undefined,
  edge: 'start' | 'end',
  form: WorkflowForm
): CollapsibleFormElement | null => {
  if (!element) {
    return null;
  }
  if (element.type === 'collapsible') {
    return element;
  }
  if (element.type !== 'container') {
    return null;
  }
  // Layout-only wrappers do not break visual adjacency; multi-item rows have no single vertical edge.
  const children = getFormChildren(form, element.id);
  if (element.data.layout === 'row' && children.length !== 1) {
    return null;
  }
  return getEdgeSection(edge === 'start' ? children[0] : children.at(-1), edge, form);
};

const ViewElement = ({
  context,
  continuesToSection,
  element,
  isAdjacentSection,
  isPreviousSectionOpen,
  isRoot,
  layout,
}: {
  context: FormViewContext;
  continuesToSection: boolean;
  element: WorkflowFormElement;
  isAdjacentSection: boolean;
  isPreviousSectionOpen: boolean;
  isRoot: boolean;
  layout: FormLayout;
}) => {
  switch (element.type) {
    case 'collapsible':
      return (
        <FormSection
          context={context}
          continuesToSection={continuesToSection}
          element={element}
          isAdjacentSection={isAdjacentSection}
          isPreviousSectionOpen={isPreviousSectionOpen}
          layout={layout}
        />
      );
    case 'container': {
      const children = getFormChildren(context.projectGraph.form, element.id);
      // A row with one item has the same geometry as a column; inherit `layout` so a wrapped section keeps full width
      // and column hairlines instead of losing borders as a row item.
      const childLayout = element.data.layout === 'row' && children.length === 1 ? layout : element.data.layout;
      const forwardsColumnEdges = layout === 'column' && childLayout === 'column';

      return (
        <Stack
          direction={childLayout}
          gap="2.5"
          minW="0"
          mt={layout === 'column' && isAdjacentSection ? '-2.5' : undefined}
        >
          <ViewChildren
            context={context}
            elements={children}
            hasNextSection={forwardsColumnEdges && continuesToSection}
            layout={childLayout}
            prevSectionOpen={forwardsColumnEdges && isPreviousSectionOpen}
          />
        </Stack>
      );
    }
    case 'node-field':
      return <NodeFieldControl element={element} projectGraph={context.projectGraph} />;
    case 'heading':
      return (
        <Text
          css={isRoot ? ROOT_HEADING_STYLES : undefined}
          fontSize="xs"
          fontWeight="semibold"
          letterSpacing="wider"
          textTransform="uppercase"
        >
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
      return <Separator data-form-divider mx={isRoot ? `-${ROOT_INSET}` : undefined} />;
  }
};

const FormSection = ({
  context,
  continuesToSection,
  element,
  isAdjacentSection,
  isPreviousSectionOpen,
  layout,
}: {
  context: FormViewContext;
  continuesToSection: boolean;
  element: CollapsibleFormElement;
  isAdjacentSection: boolean;
  isPreviousSectionOpen: boolean;
  layout: FormLayout;
}) => {
  const { t } = useTranslation();
  const children = getFormChildren(context.projectGraph.form, element.id);
  const open = context.openSections[element.id] ?? true;
  const hasInvalidField = sectionHasInvalidField(children, context.projectGraph, context.invalidFieldIds);
  const onOpenChange = useCallback(
    ({ open: nextOpen }: { open: boolean }) => context.onSectionOpenChange(element.id, nextOpen),
    [context, element.id]
  );

  return (
    <Collapsible.Root
      borderBottomWidth={layout === 'column' && !continuesToSection ? '1px' : undefined}
      borderTopWidth={layout === 'column' ? '1px' : undefined}
      className="workflow-form-section"
      css={layout === 'column' ? SECTION_STYLES : undefined}
      data-prev-open={isPreviousSectionOpen ? '' : undefined}
      flex={layout === 'row' ? '1' : undefined}
      minW="0"
      mt={layout === 'column' && isAdjacentSection ? '-2.5' : undefined}
      open={open}
      transition="border-color var(--wb-motion-duration-medium)"
      onOpenChange={onOpenChange}
    >
      <Collapsible.Trigger
        alignItems="center"
        bg="bg.subtle"
        color="fg.muted"
        css={TRIGGER_STYLES}
        display="flex"
        gap={2}
        pb="1px"
        pt="1px"
        px="0.5"
        textAlign="start"
        w="full"
      >
        <Flex
          alignItems="center"
          as="span"
          data-issues={hasInvalidField ? '' : undefined}
          data-section-chip=""
          gap={2}
          h="9"
          minW="0"
          pe="2"
          ps="1.5"
          rounded="control"
          transition="background var(--wb-motion-duration-fast), color var(--wb-motion-duration-fast)"
          w="full"
        >
          <Text
            alignItems="center"
            as="span"
            display="inline-flex"
            flex="1"
            fontSize="md"
            fontWeight="medium"
            gap="1"
            minW="0"
          >
            <Collapsible.Indicator
              _open={OPEN_INDICATOR_STYLES}
              flexShrink={0}
              transition="transform var(--wb-motion-duration-slow)"
            >
              <ChevronRightIcon size="14" />
            </Collapsible.Indicator>
            <Text as="span" data-section-label="" truncate>
              {element.data.content || t('widgets.workflow.formView.unnamedSection')}
            </Text>
            {hasInvalidField ? <Icon as={CircleAlertIcon} aria-hidden boxSize="3" flexShrink={0} /> : null}
          </Text>
          {!open && hasInvalidField ? (
            <Text as="span" color="fg.error" flexShrink="0" fontSize="xs">
              {t('widgets.workflow.formView.needsAttention')}
            </Text>
          ) : null}
        </Flex>
      </Collapsible.Trigger>
      <Collapsible.Content>
        <Box css={FIELDS_STYLES} px="0.5">
          <Stack bg={TRAY_BG} borderTopRadius={SEAM_RADIUS} direction="column" gap="2.5" minW="0" p="2" rounded="l2">
            <ViewChildren context={context} elements={children} layout="column" />
          </Stack>
        </Box>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};
