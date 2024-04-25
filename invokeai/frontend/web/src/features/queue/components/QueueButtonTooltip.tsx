import { Divider, Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import { useBoardName } from 'services/api/hooks/useBoardName';

const selectPromptsCount = createSelector(
  selectRegionalPromptsSlice,
  selectDynamicPromptsSlice,
  (regionalPrompts, dynamicPrompts) =>
    getShouldProcessPrompt(regionalPrompts.present.baseLayer.positivePrompt) ? dynamicPrompts.prompts.length : 1
);

type Props = {
  prepend?: boolean;
};

export const QueueButtonTooltip = memo(({ prepend = false }: Props) => {
  const { t } = useTranslation();
  const { isReady, reasons } = useIsReadyToEnqueue();
  const isLoadingDynamicPrompts = useAppSelector((s) => s.dynamicPrompts.isLoading);
  const promptsCount = useAppSelector(selectPromptsCount);
  const iterations = useAppSelector((s) => s.generation.iterations);
  const autoAddBoardId = useAppSelector((s) => s.gallery.autoAddBoardId);
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
      <Text fontWeight="semibold">{label}</Text>
      <Text>
        {t('queue.queueCountPrediction', {
          promptsCount,
          iterations,
          count: Math.min(promptsCount * iterations, 10000),
        })}
      </Text>
      {reasons.length > 0 && (
        <>
          <Divider opacity={0.2} borderColor="base.900" />
          <UnorderedList>
            {reasons.map((reason, i) => (
              <ListItem key={`${reason}.${i}`}>
                <Text>{reason}</Text>
              </ListItem>
            ))}
          </UnorderedList>
        </>
      )}
      <Divider opacity={0.2} borderColor="base.900" />
      <Text fontStyle="oblique 10deg">
        {t('parameters.invoke.addingImagesTo')}{' '}
        <Text as="span" fontWeight="semibold">
          {autoAddBoardName || t('boards.uncategorized')}
        </Text>
      </Text>
    </Flex>
  );
});

QueueButtonTooltip.displayName = 'QueueButtonTooltip';
