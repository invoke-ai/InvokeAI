import { Divider, Flex, ListItem, Text, UnorderedList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import { useBoardName } from 'services/api/hooks/useBoardName';
import { usePredictedQueueCounts } from '../hooks/usePredictedQueueCounts';

const tooltipSelector = createSelector(
  [stateSelector],
  ({ gallery }) => {
    const { autoAddBoardId } = gallery;
    return {
      autoAddBoardId,
    };
  },
  defaultSelectorOptions
);

type Props = {
  prepend?: boolean;
};

const QueueButtonTooltipContent = ({ prepend = false }: Props) => {
  const { t } = useTranslation();
  const { isReady, reasons } = useIsReadyToEnqueue();
  const { autoAddBoardId } = useAppSelector(tooltipSelector);
  const autoAddBoardName = useBoardName(autoAddBoardId);
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const counts = usePredictedQueueCounts();

  const label = useMemo(() => {
    if (isLoading) {
      return t('queue.enqueueing');
    }
    if (isReady) {
      if (prepend) {
        return t('queue.queueFront', { predicted: counts?.predicted ?? '?' });
      }
      return t('queue.queueBack', { predicted: counts?.predicted ?? '?' });
    }
    return t('queue.notReady');
  }, [counts?.predicted, isLoading, isReady, prepend, t]);

  return (
    <Flex flexDir="column" gap={1}>
      <Text fontWeight={600}>{label}</Text>
      {reasons.length > 0 && (
        <UnorderedList>
          {reasons.map((reason, i) => (
            <ListItem key={`${reason}.${i}`}>
              <Text fontWeight={400}>{reason}</Text>
            </ListItem>
          ))}
        </UnorderedList>
      )}
      <StyledDivider />
      <Text fontWeight={400} fontStyle="oblique 10deg">
        Adding images to{' '}
        <Text as="span" fontWeight={600}>
          {autoAddBoardName || 'Uncategorized'}
        </Text>
      </Text>
    </Flex>
  );
};

export default memo(QueueButtonTooltipContent);

const StyledDivider = memo(() => (
  <Divider
    opacity={0.2}
    borderColor="base.50"
    _dark={{ borderColor: 'base.900' }}
  />
));

StyledDivider.displayName = 'StyledDivider';
