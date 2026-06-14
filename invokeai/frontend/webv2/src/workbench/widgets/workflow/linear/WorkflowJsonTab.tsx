import { Stack } from '@chakra-ui/react';

import { JsonPreview } from '../../../components/ui/JsonPreview';
import type { ProjectGraphState } from '../../../workflows/types';
import { getWorkflowJsonText } from '../workflowTransfer';

/** Read-only view of the serialized workflow — the legacy "JSON" tab. */
export const WorkflowJsonTab = ({ projectGraph }: { projectGraph: ProjectGraphState }) => (
  <Stack p="1.5">
    <JsonPreview label="Workflow JSON" text={getWorkflowJsonText(projectGraph)} />
  </Stack>
);
