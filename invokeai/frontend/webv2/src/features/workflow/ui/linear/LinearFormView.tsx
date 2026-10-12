import type { ProjectGraphState, WorkflowFormElement } from '@features/workflow/contracts';

import { Separator, Stack, Text } from '@chakra-ui/react';
import { useWorkflowHostCommands } from '@features/workflow/ui/WorkflowUiContext';
import { getFormChildren } from '@features/workflow/utility';
import { Button } from '@platform/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { NodeFieldControl } from './NodeFieldControl';

const ViewElement = ({ element, projectGraph }: { element: WorkflowFormElement; projectGraph: ProjectGraphState }) => {
  switch (element.type) {
    case 'container':
      return (
        <Stack direction={element.data.layout === 'row' ? 'row' : 'column'} gap="3">
          {getFormChildren(projectGraph.form, element.id).map((child) => (
            <ViewElement key={child.id} element={child} projectGraph={projectGraph} />
          ))}
        </Stack>
      );
    case 'node-field':
      return <NodeFieldControl element={element} projectGraph={projectGraph} />;
    case 'heading':
      return (
        <Text fontSize="lg" fontWeight="700">
          {element.data.content}
        </Text>
      );
    case 'text':
      return (
        <Text color="fg.muted" fontSize="xs" whiteSpace="pre-wrap">
          {element.data.content}
        </Text>
      );
    case 'divider':
      return <Separator borderColor="border.subtle" />;
  }
};

export const LinearFormView = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const { t } = useTranslation();
  const { widgets } = useWorkflowHostCommands();
  const rootChildren = getFormChildren(projectGraph.form);
  const onOpenWorkflowEditorClick = useCallback(
    () => widgets.open({ region: 'center', widgetId: 'workflow' }),
    [widgets]
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
    <Stack gap="3" p="3">
      {rootChildren.map((element) => (
        <ViewElement key={element.id} element={element} projectGraph={projectGraph} />
      ))}
    </Stack>
  );
};
