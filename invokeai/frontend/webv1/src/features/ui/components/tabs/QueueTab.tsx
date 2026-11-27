import { Flex } from '@invoke-ai/ui-library';
import QueueTabContent from 'features/queue/components/QueueTabContent';
import { memo } from 'react';

const QueueTab = () => {
  return (
    <Flex layerStyle="body" w="full" h="full" p={2}>
      <QueueTabContent />
    </Flex>
  );
};

export default memo(QueueTab);
