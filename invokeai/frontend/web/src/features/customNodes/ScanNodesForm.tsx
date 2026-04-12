import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useListCustomNodePacksQuery } from 'services/api/endpoints/customNodes';

export const ScanNodesForm = memo(() => {
  const { t } = useTranslation();
  const { data } = useListCustomNodePacksQuery();

  const nodesPath = data?.node_packs?.[0]?.path;
  // Extract parent directory from a pack path, or show a placeholder
  const nodesDir = nodesPath ? nodesPath.substring(0, nodesPath.lastIndexOf('/')) : null;

  return (
    <Flex flexDir="column" gap={4} pt={4}>
      <Text fontSize="sm" color="base.500">
        {t('customNodes.scanFolderDescription')}
      </Text>
      {nodesDir && (
        <Text fontSize="xs" color="base.600">
          {t('customNodes.nodesDirectory')}: {nodesDir}
        </Text>
      )}
    </Flex>
  );
});

ScanNodesForm.displayName = 'ScanNodesForm';
