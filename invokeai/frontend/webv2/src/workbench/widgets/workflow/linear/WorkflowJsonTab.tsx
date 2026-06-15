import { Flex } from '@chakra-ui/react';

import { JsonPreview } from '@workbench/components/ui/JsonPreview';
import type { ProjectGraphState } from '@workbench/workflows/types';
import { getWorkflowJsonText } from '@workbench/widgets/workflow/workflowTransfer';

/** Read-only view of the serialized workflow — the legacy "JSON" tab. */
export const WorkflowJsonTab = ({ projectGraph }: { projectGraph: ProjectGraphState }) => (
  <Flex h="full" minH="0" minW="0" borderWidth={1} rounded="md" m="1" overflow="hidden">
    <JsonPreview label="Workflow JSON" maxH="100%" text={getWorkflowJsonText(projectGraph)} />
  </Flex>
);
