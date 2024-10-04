import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { $builtWorkflow } from 'features/nodes/hooks/useWorkflowWatcher';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowJSONTab = () => {
  const workflow = useStore($builtWorkflow);
  const { t } = useTranslation();

  return (
    <Flex flexDir="column" alignItems="flex-start" gap={2} h="full">
      <DataViewer data={workflow ?? {}} label={t('nodes.workflow')} />
    </Flex>
  );
};

export default memo(WorkflowJSONTab);
