import { Flex } from '@invoke-ai/ui-library';
import { CustomNodesInstallPane } from 'features/customNodes/CustomNodesInstallPane';
import { CustomNodesList } from 'features/customNodes/CustomNodesList';
import { memo } from 'react';

const CustomNodesManagerTab = () => {
  return (
    <Flex layerStyle="body" w="full" h="full" gap="2" p={2}>
      <CustomNodesList />
      <CustomNodesInstallPane />
    </Flex>
  );
};

export default memo(CustomNodesManagerTab);
