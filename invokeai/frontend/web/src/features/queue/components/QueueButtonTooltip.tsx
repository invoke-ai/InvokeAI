import type { TooltipProps } from '@invoke-ai/ui-library';
import { Divider, Flex, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { selectSendToCanvas } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectIterations, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import {
  selectDynamicPromptsIsLoading,
  selectDynamicPromptsSlice,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import { useBoardName } from 'services/api/hooks/useBoardName';

const selectPromptsCount = createSelector(selectParamsSlice, selectDynamicPromptsSlice, (params, dynamicPrompts) =>
  getShouldProcessPrompt(params.positivePrompt) ? dynamicPrompts.prompts.length : 1
);

type Props = TooltipProps & {
  prepend?: boolean;
};

export const QueueButtonTooltip = ({ prepend, children, ...rest }: PropsWithChildren<Props>) => {
  return (
    <Tooltip label={<TooltipContent prepend={prepend} />} maxW={512} {...rest}>
      {children}
    </Tooltip>
  );
};

const TooltipContent = memo(({ prepend = false }: { prepend?: boolean }) => {
  const { t } = useTranslation();
  const { isReady, reasons } = useIsReadyToEnqueue();
  const sendToCanvas = useAppSelector(selectSendToCanvas);
  const isLoadingDynamicPrompts = useAppSelector(selectDynamicPromptsIsLoading);
  const promptsCount = useAppSelector(selectPromptsCount);
  const iterationsCount = useAppSelector(selectIterations);
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAddBoardName = useBoardName(autoAddBoardId);
  const [_, { isLoading }] = useEnqueueBatchMutation({
    fixedCacheKey: 'enqueueBatch',
  });
  const queueCountPredictionLabel = useMemo(() => {
    const generationCount = Math.min(promptsCount * iterationsCount, 10000);
    const prompts = t('queue.prompts', { count: promptsCount });
    const iterations = t('queue.iterations', { count: iterationsCount });
    const generations = t('queue.generations', { count: generationCount });
    return `${promptsCount} ${prompts} \u00d7 ${iterationsCount} ${iterations} -> ${generationCount} ${generations}`.toLowerCase();
  }, [iterationsCount, promptsCount, t]);

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

  const addingTo = useMemo(() => {
    if (sendToCanvas) {
      return t('controlLayers.stagingOnCanvas');
    }
    return t('parameters.invoke.addingImagesTo');
  }, [sendToCanvas, t]);

  const destination = useMemo(() => {
    if (sendToCanvas) {
      return t('queue.canvas');
    }
    if (autoAddBoardName) {
      return autoAddBoardName;
    }
    return t('boards.uncategorized');
  }, [autoAddBoardName, sendToCanvas, t]);

  return (
    <Flex flexDir="column" gap={1}>
      <Text fontWeight="semibold">{label}</Text>
      <Text>{queueCountPredictionLabel}</Text>
      {reasons.length > 0 && (
        <>
          <Divider opacity={0.2} borderColor="base.900" />
          <UnorderedList>
            {reasons.map((reason, i) => (
              <ListItem key={`${reason.content}.${i}`}>
                <span>
                  {reason.prefix && (
                    <Text as="span" fontWeight="semibold">
                      {reason.prefix}:{' '}
                    </Text>
                  )}
                  <Text as="span">{reason.content}</Text>
                </span>
              </ListItem>
            ))}
          </UnorderedList>
        </>
      )}
      <Divider opacity={0.2} borderColor="base.900" />
      <Text fontStyle="oblique 10deg">
        {addingTo}{' '}
        <Text as="span" fontWeight="semibold">
          {destination}
        </Text>
      </Text>
    </Flex>
  );
});

TooltipContent.displayName = 'QueueButtonTooltipContent';
