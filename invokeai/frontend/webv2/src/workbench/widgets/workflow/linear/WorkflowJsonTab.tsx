import type { ProjectGraphState } from '@workbench/workflows/types';

import { Flex } from '@chakra-ui/react';
import { JsonPreview } from '@workbench/components/ui';
import { getWorkflowJsonText } from '@workbench/widgets/workflow/workflowTransfer';
import { useTranslation } from 'react-i18next';

/** Read-only view of the serialized workflow — the legacy "JSON" tab. */
export const WorkflowJsonTab = ({ projectGraph }: { projectGraph: ProjectGraphState }) => {
  const { t } = useTranslation();

  return (
    <Flex h="full" minH="0" minW="0" borderWidth={1} rounded="md" m="1" overflow="hidden">
      <JsonPreview label={t('widgets.workflow.workflowJson')} maxH="100%" text={getWorkflowJsonText(projectGraph)} />
    </Flex>
  );
};
