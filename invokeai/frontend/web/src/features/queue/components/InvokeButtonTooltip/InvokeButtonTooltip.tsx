import type { TooltipProps } from '@invoke-ai/ui-library';
import { Divider, Flex, ListItem, Text, Tooltip, UnorderedList } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $true } from 'app/store/nanostores/util';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectSendToCanvas } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectIterations } from 'features/controlLayers/store/paramsSlice';
import { selectDynamicPromptsIsLoading } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectAutoAddBoardId } from 'features/gallery/store/gallerySelectors';
import { $templates } from 'features/nodes/store/nodesSlice';
import type { Reason } from 'features/queue/store/readiness';
import {
  buildSelectIsReadyToEnqueueCanvasTab,
  buildSelectIsReadyToEnqueueUpscaleTab,
  buildSelectIsReadyToEnqueueWorkflowsTab,
  buildSelectReasonsWhyCannotEnqueueCanvasTab,
  buildSelectReasonsWhyCannotEnqueueUpscaleTab,
  buildSelectReasonsWhyCannotEnqueueWorkflowsTab,
  selectPromptsCount,
  selectWorkflowsBatchSize,
} from 'features/queue/store/readiness';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { enqueueMutationFixedCacheKeyOptions, useEnqueueBatchMutation } from 'services/api/endpoints/queue';
import { useBoardName } from 'services/api/hooks/useBoardName';
import { $isConnected } from 'services/events/stores';

type Props = TooltipProps & {
  prepend?: boolean;
};

export const InvokeButtonTooltip = ({ prepend, children, ...rest }: PropsWithChildren<Props>) => {
  return (
    <Tooltip label={<TooltipContent prepend={prepend} />} maxW={512} {...rest}>
      {children}
    </Tooltip>
  );
};

const TooltipContent = memo(({ prepend = false }: { prepend?: boolean }) => {
  const activeTab = useAppSelector(selectActiveTab);

  if (activeTab === 'canvas') {
    return <CanvasTabTooltipContent prepend={prepend} />;
  }

  if (activeTab === 'workflows') {
    return <WorkflowsTabTooltipContent prepend={prepend} />;
  }

  if (activeTab === 'upscaling') {
    return <UpscaleTabTooltipContent prepend={prepend} />;
  }

  return null;
});
TooltipContent.displayName = 'TooltipContent';

const CanvasTabTooltipContent = memo(({ prepend = false }: { prepend?: boolean }) => {
  const isConnected = useStore($isConnected);
  const canvasManager = useCanvasManagerSafe();
  const canvasIsFiltering = useStore(canvasManager?.stateApi.$isFiltering ?? $true);
  const canvasIsTransforming = useStore(canvasManager?.stateApi.$isTransforming ?? $true);
  const canvasIsRasterizing = useStore(canvasManager?.stateApi.$isRasterizing ?? $true);
  const canvasIsSelectingObject = useStore(canvasManager?.stateApi.$isSegmenting ?? $true);
  const canvasIsCompositing = useStore(canvasManager?.compositor.$isBusy ?? $true);

  const selectIsReady = useMemo(
    () =>
      buildSelectIsReadyToEnqueueCanvasTab({
        isConnected,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsSelectingObject,
        canvasIsCompositing,
      }),
    [
      isConnected,
      canvasIsCompositing,
      canvasIsFiltering,
      canvasIsRasterizing,
      canvasIsSelectingObject,
      canvasIsTransforming,
    ]
  );

  const selectReasons = useMemo(
    () =>
      buildSelectReasonsWhyCannotEnqueueCanvasTab({
        isConnected,
        canvasIsFiltering,
        canvasIsTransforming,
        canvasIsRasterizing,
        canvasIsSelectingObject,
        canvasIsCompositing,
      }),
    [
      isConnected,
      canvasIsCompositing,
      canvasIsFiltering,
      canvasIsRasterizing,
      canvasIsSelectingObject,
      canvasIsTransforming,
    ]
  );

  const isReady = useAppSelector(selectIsReady);
  const reasons = useAppSelector(selectReasons);

  return (
    <Flex flexDir="column" gap={1}>
      <IsReadyText isReady={isReady} prepend={prepend} />
      <QueueCountPredictionCanvasOrUpscaleTab />
      {reasons.length > 0 && (
        <>
          <StyledDivider />
          <ReasonsList reasons={reasons} />
        </>
      )}
      <StyledDivider />
      <AddingToText />
    </Flex>
  );
});
CanvasTabTooltipContent.displayName = 'CanvasTabTooltipContent';

const UpscaleTabTooltipContent = memo(({ prepend = false }: { prepend?: boolean }) => {
  const isConnected = useStore($isConnected);

  const selectIsReady = useMemo(() => buildSelectIsReadyToEnqueueUpscaleTab({ isConnected }), [isConnected]);
  const selectReasons = useMemo(() => buildSelectReasonsWhyCannotEnqueueUpscaleTab({ isConnected }), [isConnected]);

  const isReady = useAppSelector(selectIsReady);
  const reasons = useAppSelector(selectReasons);

  return (
    <Flex flexDir="column" gap={1}>
      <IsReadyText isReady={isReady} prepend={prepend} />
      <QueueCountPredictionCanvasOrUpscaleTab />
      {reasons.length > 0 && (
        <>
          <StyledDivider />
          <ReasonsList reasons={reasons} />
        </>
      )}
    </Flex>
  );
});
UpscaleTabTooltipContent.displayName = 'UpscaleTabTooltipContent';

const WorkflowsTabTooltipContent = memo(({ prepend = false }: { prepend?: boolean }) => {
  const isConnected = useStore($isConnected);
  const templates = useStore($templates);

  const selectIsReady = useMemo(
    () => buildSelectIsReadyToEnqueueWorkflowsTab({ isConnected, templates }),
    [isConnected, templates]
  );
  const selectReasons = useMemo(
    () => buildSelectReasonsWhyCannotEnqueueWorkflowsTab({ isConnected, templates }),
    [isConnected, templates]
  );

  const isReady = useAppSelector(selectIsReady);
  const reasons = useAppSelector(selectReasons);

  return (
    <Flex flexDir="column" gap={1}>
      <IsReadyText isReady={isReady} prepend={prepend} />
      <QueueCountPredictionWorkflowsTab />
      {reasons.length > 0 && (
        <>
          <StyledDivider />
          <ReasonsList reasons={reasons} />
        </>
      )}
    </Flex>
  );
});
WorkflowsTabTooltipContent.displayName = 'WorkflowsTabTooltipContent';

const QueueCountPredictionCanvasOrUpscaleTab = memo(() => {
  const { t } = useTranslation();
  const promptsCount = useAppSelector(selectPromptsCount);
  const iterationsCount = useAppSelector(selectIterations);

  const text = useMemo(() => {
    const generationCount = Math.min(promptsCount * iterationsCount, 10000);
    const prompts = t('queue.prompts', { count: promptsCount });
    const iterations = t('queue.iterations', { count: iterationsCount });
    const generations = t('queue.generations', { count: generationCount });
    return `${promptsCount} ${prompts} \u00d7 ${iterationsCount} ${iterations} -> ${generationCount} ${generations}`.toLowerCase();
  }, [iterationsCount, promptsCount, t]);

  return <Text>{text}</Text>;
});
QueueCountPredictionCanvasOrUpscaleTab.displayName = 'QueueCountPredictionCanvasOrUpscaleTab';

const QueueCountPredictionWorkflowsTab = memo(() => {
  const { t } = useTranslation();
  const batchSize = useAppSelector(selectWorkflowsBatchSize);
  const iterationsCount = useAppSelector(selectIterations);

  const text = useMemo(() => {
    const generationCount = Math.min(batchSize * iterationsCount, 10000);
    const iterations = t('queue.iterations', { count: iterationsCount });
    const generations = t('queue.generations', { count: generationCount });
    return `${batchSize} ${t('queue.batchSize')} \u00d7 ${iterationsCount} ${iterations} -> ${generationCount} ${generations}`.toLowerCase();
  }, [batchSize, iterationsCount, t]);

  return <Text>{text}</Text>;
});
QueueCountPredictionWorkflowsTab.displayName = 'QueueCountPredictionWorkflowsTab';

const IsReadyText = memo(({ isReady, prepend }: { isReady: boolean; prepend: boolean }) => {
  const { t } = useTranslation();
  const isLoadingDynamicPrompts = useAppSelector(selectDynamicPromptsIsLoading);
  const [_, enqueueMutation] = useEnqueueBatchMutation(enqueueMutationFixedCacheKeyOptions);

  const text = useMemo(() => {
    if (enqueueMutation.isLoading) {
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
  }, [enqueueMutation.isLoading, isLoadingDynamicPrompts, isReady, prepend, t]);

  return <Text fontWeight="semibold">{text}</Text>;
});
IsReadyText.displayName = 'IsReadyText';

const ReasonsList = memo(({ reasons }: { reasons: Reason[] }) => {
  return (
    <UnorderedList>
      {reasons.map((reason, i) => (
        <ReasonListItem key={`${reason.content}.${i}`} reason={reason} />
      ))}
    </UnorderedList>
  );
});
ReasonsList.displayName = 'ReasonsList';

const ReasonListItem = memo(({ reason }: { reason: Reason }) => {
  return (
    <ListItem>
      <span>
        {reason.prefix && (
          <Text as="span" fontWeight="semibold">
            {reason.prefix}:{' '}
          </Text>
        )}
        <Text as="span">{reason.content}</Text>
      </span>
    </ListItem>
  );
});
ReasonListItem.displayName = 'ReasonListItem';

const StyledDivider = memo(() => <Divider opacity={0.2} borderColor="base.900" />);
StyledDivider.displayName = 'StyledDivider';

const AddingToText = memo(() => {
  const { t } = useTranslation();
  const sendToCanvas = useAppSelector(selectSendToCanvas);
  const autoAddBoardId = useAppSelector(selectAutoAddBoardId);
  const autoAddBoardName = useBoardName(autoAddBoardId);

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
    <Text fontStyle="oblique 10deg">
      {addingTo}{' '}
      <Text as="span" fontWeight="semibold">
        {destination}
      </Text>
    </Text>
  );
});
AddingToText.displayName = 'AddingToText';
