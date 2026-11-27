import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Button, Flex, Heading, Spinner, Text } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold } from 'react-icons/pi';
import {
  useListCustomNodePacksQuery,
  useReloadCustomNodesMutation,
  useUninstallCustomNodePackMutation,
} from 'services/api/endpoints/customNodes';

const listSx: SystemStyleObject = {
  flexDir: 'column',
  p: 4,
  gap: 4,
  borderRadius: 'base',
  w: '50%',
  minWidth: '360px',
  h: 'full',
};

type NodePackInfo = {
  name: string;
  path: string;
  node_count: number;
  node_types: string[];
};

const NodePackItem = memo(({ pack }: { pack: NodePackInfo }) => {
  const { t } = useTranslation();
  const [uninstallPack] = useUninstallCustomNodePackMutation();

  const handleUninstall = useCallback(() => {
    uninstallPack(pack.name);
  }, [uninstallPack, pack.name]);

  return (
    <Flex layerStyle="first" p={4} borderRadius="base" flexDir="column" gap={2}>
      <Flex justifyContent="space-between" alignItems="center">
        <Heading fontSize="md">{pack.name}</Heading>
        <Button size="xs" colorScheme="error" onClick={handleUninstall}>
          {t('customNodes.uninstall')}
        </Button>
      </Flex>
      <Flex gap={2} alignItems="center" flexWrap="wrap">
        <Badge colorScheme="invokeBlue">{t('customNodes.nodeCount', { count: pack.node_count })}</Badge>
        {pack.node_types.map((nodeType) => (
          <Badge key={nodeType} variant="subtle">
            {nodeType}
          </Badge>
        ))}
      </Flex>
      <Text fontSize="xs" color="base.500" noOfLines={1}>
        {pack.path}
      </Text>
    </Flex>
  );
});

NodePackItem.displayName = 'NodePackItem';

export const CustomNodesList = memo(() => {
  const { t } = useTranslation();
  const { data, isLoading } = useListCustomNodePacksQuery();
  const [reloadNodes, { isLoading: isReloading }] = useReloadCustomNodesMutation();

  const handleReload = useCallback(() => {
    reloadNodes();
  }, [reloadNodes]);

  return (
    <Flex sx={listSx}>
      <Flex w="full" gap={4} justifyContent="space-between" alignItems="center">
        <Heading fontSize="xl" py={1}>
          {t('customNodes.title')}
        </Heading>
        <Button
          size="sm"
          leftIcon={<PiArrowClockwiseBold />}
          onClick={handleReload}
          isLoading={isReloading}
          loadingText={t('customNodes.reloading')}
        >
          {t('customNodes.reload')}
        </Button>
      </Flex>

      <Flex flexDir="column" gap={3} w="full" h="full" overflow="auto">
        {isLoading && (
          <Flex justifyContent="center" py={8}>
            <Spinner />
          </Flex>
        )}

        {data && data.node_packs.length === 0 && (
          <Flex justifyContent="center" py={8}>
            <Text color="base.500">{t('customNodes.noNodePacks')}</Text>
          </Flex>
        )}

        {data?.node_packs.map((pack) => (
          <NodePackItem key={pack.name} pack={pack} />
        ))}
      </Flex>
    </Flex>
  );
});

CustomNodesList.displayName = 'CustomNodesList';
