import type { ProjectGraphState, WorkflowFormElement } from '@features/workflow/contracts';

import { Separator, Stack, Text } from '@chakra-ui/react';
import { useWorkflowHostCommands } from '@features/workflow/ui/WorkflowUiContext';
import { getFormChildren } from '@features/workflow/utility';
import { Button } from '@platform/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { LinkifiedText } from './LinkifiedText';
import { NodeFieldControl } from './NodeFieldControl';

/**
 * The form reads like the Generate and Upscale sections: a heading is a section title, and a hairline that runs edge
 * to edge starts a section. Only root elements bleed; inside a container a line stays within its column.
 */
const ROOT_INSET = '3';
const ROOT_HEADING_STYLES = {
  '&:not(:first-child)': { borderTopWidth: '1px', mx: `-${ROOT_INSET}`, pt: ROOT_INSET, px: ROOT_INSET },
  // An authored divider already drew the line. With equal specificity, this must follow the rule above.
  '[data-form-divider] + &': { borderTopWidth: 0, mx: 0, pt: 0, px: 0 },
} as const;

const ViewElement = ({
  element,
  isRoot = false,
  projectGraph,
}: {
  element: WorkflowFormElement;
  isRoot?: boolean;
  projectGraph: ProjectGraphState;
}) => {
  switch (element.type) {
    case 'container':
      return (
        <Stack direction={element.data.layout === 'row' ? 'row' : 'column'} gap="2.5">
          {getFormChildren(projectGraph.form, element.id).map((child) => (
            <ViewElement key={child.id} element={child} projectGraph={projectGraph} />
          ))}
        </Stack>
      );
    case 'node-field':
      return <NodeFieldControl element={element} projectGraph={projectGraph} />;
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
    <Stack gap="2.5" p={ROOT_INSET}>
      {rootChildren.map((element) => (
        <ViewElement key={element.id} element={element} isRoot projectGraph={projectGraph} />
      ))}
    </Stack>
  );
};
