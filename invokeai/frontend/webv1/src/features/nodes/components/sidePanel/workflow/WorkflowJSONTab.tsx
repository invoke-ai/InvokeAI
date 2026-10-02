import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { $previewWorkflow } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowJSONTab = () => {
  const previewWorkflow = useStore($previewWorkflow);
  const { t } = useTranslation();

  return (
    <Flex flexDir="column" alignItems="flex-start" gap={2} h="full">
      <DataViewer data={previewWorkflow} label={t('nodes.workflow')} bg="base.850" color="base.200" />
    </Flex>
  );
};

export default memo(WorkflowJSONTab);
