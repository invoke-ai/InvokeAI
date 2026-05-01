import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListCustomNodePacksQuery } from 'services/api/endpoints/customNodes';

export const ScanNodesForm = memo(() => {
  const { t } = useTranslation();
  const { data } = useListCustomNodePacksQuery();

  return (
    <Flex flexDir="column" gap={4} pt={4}>
      <Text fontSize="sm" color="base.500">
        {t('customNodes.scanFolderDescription')}
      </Text>
      {data?.custom_nodes_path && (
        <Text fontSize="xs" color="base.600">
          {t('customNodes.nodesDirectory')}: {data.custom_nodes_path}
        </Text>
      )}
    </Flex>
  );
});

ScanNodesForm.displayName = 'ScanNodesForm';
