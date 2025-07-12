import { Box, Button, Flex, Heading } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { memo, useCallback, useMemo } from 'react';
import { useListModelInstallsQuery, usePruneCompletedModelInstallsMutation } from 'services/api/endpoints/models';

import { ModelInstallQueueItem } from './ModelInstallQueueItem';

export const ModelInstallQueue = memo(() => {
  const { data } = useListModelInstallsQuery();

  const [_pruneCompletedModelInstalls] = usePruneCompletedModelInstallsMutation();

  const pruneCompletedModelInstalls = useCallback(() => {
    _pruneCompletedModelInstalls()
      .unwrap()
      .then((_) => {
        toast({
          id: 'MODEL_INSTALL_QUEUE_PRUNED',
          title: t('toast.prunedQueue'),
          status: 'success',
        });
      })
      .catch((error) => {
        if (error) {
          toast({
            id: 'MODEL_INSTALL_QUEUE_PRUNE_FAILED',
            title: `${error.data.detail} `,
            status: 'error',
          });
        }
      });
  }, [_pruneCompletedModelInstalls]);

  const pruneAvailable = useMemo(() => {
    return data?.some(
      (model) => model.status === 'cancelled' || model.status === 'error' || model.status === 'completed'
    );
  }, [data]);

  return (
    <Flex flexDir="column" p={3} h="full" gap={3}>
      <Flex justifyContent="space-between" alignItems="center">
        <Heading size="sm">{t('modelManager.installQueue')}</Heading>
        <Button
          size="sm"
          isDisabled={!pruneAvailable}
          onClick={pruneCompletedModelInstalls}
          tooltip={t('modelManager.pruneTooltip')}
        >
          {t('modelManager.prune')}
        </Button>
      </Flex>
      <Box layerStyle="first" p={3} borderRadius="base" w="full" h="full">
        <ScrollableContent>
          <Flex flexDir="column-reverse" gap="2" w="full">
            {data?.map((model) => <ModelInstallQueueItem key={model.id} installJob={model} />)}
          </Flex>
        </ScrollableContent>
      </Box>
    </Flex>
  );
});

ModelInstallQueue.displayName = 'ModelInstallQueue';
