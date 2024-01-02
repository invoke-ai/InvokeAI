import { Divider, Flex, ListItem, UnorderedList } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvText } from 'common/components/InvText/wrapper';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import { useBoardName } from 'services/api/hooks/useBoardName';

const StyledDivider = () => <Divider opacity={0.2} borderColor="base.900" />;

const tooltipSelector = createMemoizedSelector(
  [stateSelector],
  ({ gallery, dynamicPrompts, generation }) => {
    const { autoAddBoardId } = gallery;
    const { iterations, positivePrompt } = generation;
    const promptsCount = getShouldProcessPrompt(positivePrompt)
      ? dynamicPrompts.prompts.length
      : 1;
    return {
      autoAddBoardId,
      promptsCount,
      iterations,
    };
  }
);

type Props = {
  prepend?: boolean;
};

export const QueueButtonTooltip = memo(({ prepend = false }: Props) => {
  const { t } = useTranslation();
  const { isReady, reasons } = useIsReadyToEnqueue();
  const isLoadingDynamicPrompts = useAppSelector(
    (state) => state.dynamicPrompts.isLoading
  );
  const { autoAddBoardId, promptsCount, iterations } =
    useAppSelector(tooltipSelector);
  const autoAddBoardName = useBoardName(autoAddBoardId);
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });

  const label = useMemo(() => {
    if (isLoading) {
      return t('queue.enqueueing');
    }
    if (isLoadingDynamicPrompts) {
      return t('dynamicPrompts.loading');
    }
    if (isReady) {
      if (prepend) {
        return t('queue.queueFront');
      }
      return t('queue.queueBack');
    }
    return t('queue.notReady');
  }, [isLoading, isLoadingDynamicPrompts, isReady, prepend, t]);

  return (
    <Flex flexDir="column" gap={1}>
      <InvText fontWeight="semibold">{label}</InvText>
      <InvText>
        {t('queue.queueCountPrediction', {
          promptsCount,
          iterations,
          count: Math.min(promptsCount * iterations, 10000),
        })}
      </InvText>
      {reasons.length > 0 && (
        <>
          <StyledDivider />
          <UnorderedList>
            {reasons.map((reason, i) => (
              <ListItem key={`${reason}.${i}`}>
                <InvText>{reason}</InvText>
              </ListItem>
            ))}
          </UnorderedList>
        </>
      )}
      <StyledDivider />
      <InvText fontStyle="oblique 10deg">
        {t('parameters.invoke.addingImagesTo')}{' '}
        <InvText as="span" fontWeight="semibold">
          {autoAddBoardName || t('boards.uncategorized')}
        </InvText>
      </InvText>
    </Flex>
  );
});

QueueButtonTooltip.displayName = 'QueueButtonTooltip';
