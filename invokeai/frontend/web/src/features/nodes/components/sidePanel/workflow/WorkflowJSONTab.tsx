import { Flex } from '@chakra-ui/react';
import DataViewer from 'features/gallery/components/ImageMetadataViewer/DataViewer';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const WorkflowJSONTab = () => {
  const workflow = useWorkflow();
  const { t } = useTranslation();

  return (
    <Flex
      sx={{
        flexDir: 'column',
        alignItems: 'flex-start',
        gap: 2,
        h: 'full',
      }}
    >
      <DataViewer data={workflow} label={t('nodes.workflow')} />
    </Flex>
  );
};

export default memo(WorkflowJSONTab);
