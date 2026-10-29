import type { ProjectGraphState } from '@features/workflow/contracts';

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

import {
  getFormSectionOpenState,
  groupFormSections,
  updateFormSectionOpenState,
  type FormSectionItem,
} from './formSections';
import { getInvalidFormFieldIds, sectionHasInvalidField } from './formSectionStatus';
import { LinkifiedText } from './LinkifiedText';
import { NodeFieldControl } from './NodeFieldControl';

type FormViewContext = {
  projectGraph: ProjectGraphState;
  sectionGapPx: number;
  openSections: Readonly<Record<string, boolean>>;
  invalidFieldIds: ReadonlySet<string>;
  onSectionOpenChange: (sectionId: string, open: boolean) => void;
};

const OPEN_INDICATOR_STYLES = { transform: 'rotate(90deg)' };
const DEFAULT_SECTION_GAP_PX = 16;

const ViewElement = ({
  element,
  context,
}: {
  element: Extract<FormSectionItem, { kind: 'element' }>['element'];
  context: FormViewContext;
}) => {
  switch (element.type) {
    case 'container':
      return (
        <Stack
          direction={element.data.layout === 'row' ? 'row' : 'column'}
          gap={element.data.layout === 'row' ? '3' : `${context.sectionGapPx}px`}
        >
          <ViewChildren context={context} containerId={element.id} />
        </Stack>
      );
    case 'node-field':
      return <NodeFieldControl element={element} projectGraph={context.projectGraph} />;
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

const ViewChildren = ({ context, containerId }: { context: FormViewContext; containerId?: string }) =>
  groupFormSections(getFormChildren(context.projectGraph.form, containerId)).map((item) =>
    item.kind === 'section' ? (
      <FormSection key={item.heading.id} context={context} section={item} />
    ) : (
      <ViewElement key={item.element.id} context={context} element={item.element} />
    )
  );

const FormSection = ({
  context,
  section,
}: {
  context: FormViewContext;
  section: Extract<FormSectionItem, { kind: 'section' }>;
}) => {
  const { t } = useTranslation();
  const { heading, body } = section;
  const open = context.openSections[heading.id] ?? true;
  const hasInvalidField = sectionHasInvalidField(body, context.projectGraph, context.invalidFieldIds);
  const onOpenChange = useCallback(
    ({ open: nextOpen }: { open: boolean }) => context.onSectionOpenChange(heading.id, nextOpen),
    [context, heading.id]
  );

  return (
    <Collapsible.Root
      bg={open ? 'bg.muted' : 'transparent'}
      borderColor={open ? 'border.subtle' : 'transparent'}
      borderWidth="1px"
      open={open}
      rounded="sm"
      onOpenChange={onOpenChange}
    >
      <Collapsible.Trigger alignItems="center" display="flex" gap="2" minH="8" px="2" py="1" textAlign="start" w="full">
        <Collapsible.Indicator _open={OPEN_INDICATOR_STYLES} transition="transform var(--wb-motion-duration-slow)">
          <ChevronRightIcon size="14" />
        </Collapsible.Indicator>
        <Text as="span" color="fg" flex="1" fontSize="sm" fontWeight="700" minW="0">
          {heading.data.content || t('widgets.workflow.formView.unnamedSection', { defaultValue: 'Section' })}
        </Text>
        {!open && hasInvalidField ? (
          <Text as="span" color="fg.error" fontSize="xs">
            {t('widgets.workflow.formView.needsAttention', { defaultValue: 'Needs attention' })}
          </Text>
        ) : null}
      </Collapsible.Trigger>
      <Collapsible.Content>
        <Stack borderTopColor="border.subtle" borderTopWidth="1px" gap="3" p="2">
          {body.map((element) => (
            <ViewElement key={element.id} context={context} element={element} />
          ))}
        </Stack>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

export const LinearFormView = ({
  projectGraph,
  sectionGapPx = DEFAULT_SECTION_GAP_PX,
}: {
  projectGraph: ProjectGraphState;
  sectionGapPx?: number;
}) => {
  const { t } = useTranslation();
  const { widgets } = useWorkflowHostCommands();
  const { project } = useWorkflowUi();
  const templates = useInvocationTemplatesSelector((snapshot) => snapshot.templates);
  const projectId = useWorkflowProjectSelector((snapshot) => snapshot.id);
  const storedOpenSections = useWorkflowProjectSelector((snapshot) => snapshot.workflowValues?.formSectionsOpen);
  const rootChildren = getFormChildren(projectGraph.form);
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
    () => ({ projectGraph, sectionGapPx, invalidFieldIds, openSections, onSectionOpenChange }),
    [projectGraph, sectionGapPx, invalidFieldIds, openSections, onSectionOpenChange]
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
    <Stack gap={`${sectionGapPx}px`} p="3">
      <ViewChildren context={context} />
    </Stack>
  );
};
