import { Flex } from '@chakra-ui/react';

import { JsonPreview } from '../../../components/ui/JsonPreview';
import type { ProjectGraphState } from '../../../workflows/types';
import { getWorkflowJsonText } from '../workflowTransfer';

/** Read-only view of the serialized workflow — the legacy "JSON" tab. */
export const WorkflowJsonTab = ({ projectGraph }: { projectGraph: ProjectGraphState }) => (
  <Flex h="full" minH="0" minW="0" p="1.5">
    <JsonPreview label="Workflow JSON" maxH="100%" text={getWorkflowJsonText(projectGraph)} />
  </Flex>
);
