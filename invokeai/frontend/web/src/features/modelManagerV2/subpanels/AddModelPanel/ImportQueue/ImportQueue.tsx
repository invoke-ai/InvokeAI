import { Box, Button, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { useGetModelImportsQuery, usePruneModelImportsMutation } from 'services/api/endpoints/models';

import { ImportQueueItem } from './ImportQueueItem';

export const ImportQueue = () => {
  const dispatch = useAppDispatch();

  const { data } = useGetModelImportsQuery();

  const [pruneModelImports] = usePruneModelImportsMutation();

  const pruneQueue = useCallback(() => {
    pruneModelImports()
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('toast.prunedQueue'),
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: `${error.data.detail} `,
                status: 'error',
              })
            )
          );
        }
      });
  }, [pruneModelImports, dispatch]);

  const pruneAvailable = useMemo(() => {
    return data?.some(
      (model) => model.status === 'cancelled' || model.status === 'error' || model.status === 'completed'
    );
  }, [data]);

  return (
    <Flex flexDir="column" p={3} h="full">
      <Flex justifyContent="space-between" alignItems="center">
        <Text>{t('modelManager.importQueue')}</Text>
        <Button size="sm" isDisabled={!pruneAvailable} onClick={pruneQueue} tooltip={t('modelManager.pruneTooltip')}>
          {t('modelManager.prune')}
        </Button>
      </Flex>
      <Box mt={3} layerStyle="first" p={3} borderRadius="base" w="full" h="full">
        <ScrollableContent>
          <Flex flexDir="column-reverse" gap="2">
            {data?.map((model) => <ImportQueueItem key={model.id} model={model} />)}
          </Flex>
        </ScrollableContent>
      </Box>
    </Flex>
  );
};
