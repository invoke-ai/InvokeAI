import type { ProjectGraphState, WorkflowFormElement } from '@workbench/workflows/types';

import { Separator, Stack, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui/Button';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { getFormChildren } from '@workbench/workflows/document';

import { NodeFieldControl } from './NodeFieldControl';

/**
 * The Linear UI's view mode: the form rendered as clean, runnable controls —
 * field values stay editable, the form structure does not.
 */

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
        <Text fontSize="sm" fontWeight="700">
          {element.data.content}
        </Text>
      );
    case 'text':
      return (
        <Text color="fg.muted" fontSize="2xs" whiteSpace="pre-wrap">
          {element.data.content}
        </Text>
      );
    case 'divider':
      return <Separator borderColor="border.subtle" />;
  }
};

export const LinearFormView = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const dispatch = useWorkbenchDispatch();
  const rootChildren = getFormChildren(projectGraph.form);

  if (rootChildren.length === 0) {
    return (
      <Stack gap="2" px="1" py="1">
        <Text color="fg.subtle" fontSize="2xs">
          No fields are exposed yet. Pin fields in the Workflow editor, or switch to Edit mode to build this form — it
          maps the project graph to simple controls, like the legacy Linear UI.
        </Text>
        <Button
          size="2xs"
          variant="outline"
          w="fit-content"
          onClick={() => dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: 'workflow' })}
        >
          Open Workflow Editor
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
