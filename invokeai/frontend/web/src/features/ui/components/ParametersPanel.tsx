import {
  Box,
  ButtonGroup,
  Flex,
  Spacer,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import CancelQueueButton from 'features/queue/components/CancelQueueButton';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import QueueBackButton from 'features/queue/components/QueueBackButton';
import QueueFrontButton from 'features/queue/components/QueueFrontButton';
import StartQueueButton from 'features/queue/components/StartQueueButton';
import StopQueueButton from 'features/queue/components/StopQueueButton';
import SDXLImageToImageTabParameters from 'features/sdxl/components/SDXLImageToImageTabParameters';
import SDXLTextToImageTabParameters from 'features/sdxl/components/SDXLTextToImageTabParameters';
import SDXLUnifiedCanvasTabParameters from 'features/sdxl/components/SDXLUnifiedCanvasTabParameters';
import ProgressBar from 'features/system/components/ProgressBar';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { PropsWithChildren, memo } from 'react';
import { activeTabNameSelector } from '../store/uiSelectors';
import ImageToImageTabParameters from './tabs/ImageToImage/ImageToImageTabParameters';
import TextToImageTabParameters from './tabs/TextToImage/TextToImageTabParameters';
import UnifiedCanvasParameters from './tabs/UnifiedCanvas/UnifiedCanvasParameters';
import { usePredictedQueueCounts } from 'features/queue/hooks/usePredictedQueueCounts';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const ParametersPanel = () => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const model = useAppSelector((state: RootState) => state.generation.model);

  if (activeTabName === 'txt2img') {
    return (
      <ParametersPanelWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLTextToImageTabParameters />
        ) : (
          <TextToImageTabParameters />
        )}
      </ParametersPanelWrapper>
    );
  }

  if (activeTabName === 'img2img') {
    return (
      <ParametersPanelWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLImageToImageTabParameters />
        ) : (
          <ImageToImageTabParameters />
        )}
      </ParametersPanelWrapper>
    );
  }

  if (activeTabName === 'unifiedCanvas') {
    return (
      <ParametersPanelWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLUnifiedCanvasTabParameters />
        ) : (
          <UnifiedCanvasParameters />
        )}
      </ParametersPanelWrapper>
    );
  }

  return null;
};

export default memo(ParametersPanel);

const ParametersPanelWrapper = memo((props: PropsWithChildren) => {
  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        flexDir: 'column',
        gap: 2,
      }}
    >
      <Flex
        layerStyle="first"
        sx={{
          w: 'full',
          position: 'relative',
          borderRadius: 'base',
          p: 2,
          gap: 2,
          flexDir: 'column',
        }}
      >
        <Flex h={2} w="full">
          <ProgressBar />
        </Flex>
        <Flex gap={2} w="full">
          <ButtonGroup isAttached flexGrow={2}>
            <QueueBackButton />
            <QueueFrontButton />
          </ButtonGroup>
          <ButtonGroup isAttached>
            <StartQueueButton asIconButton />
            <StopQueueButton asIconButton />
            <CancelQueueButton asIconButton />
            <ClearQueueButton asIconButton />
          </ButtonGroup>
        </Flex>
        <QueueCounts />
      </Flex>
      <Flex
        layerStyle="first"
        sx={{
          w: 'full',
          h: 'full',
          position: 'relative',
          borderRadius: 'base',
          p: 2,
        }}
      >
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            position: 'relative',
          }}
        >
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
            }}
          >
            <OverlayScrollbarsComponent
              defer
              style={{ height: '100%', width: '100%' }}
              options={{
                scrollbars: {
                  visibility: 'auto',
                  autoHide: 'scroll',
                  autoHideDelay: 800,
                  theme: 'os-theme-dark',
                },
                overflow: {
                  x: 'hidden',
                },
              }}
            >
              <Flex
                sx={{
                  gap: 2,
                  flexDirection: 'column',
                  h: 'full',
                  w: 'full',
                }}
              >
                {props.children}
              </Flex>
            </OverlayScrollbarsComponent>
          </Box>
        </Flex>
      </Flex>
    </Flex>
  );
});

ParametersPanelWrapper.displayName = 'ParametersPanelWrapper';

const QueueCounts = memo(() => {
  const counts = usePredictedQueueCounts();
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { t } = useTranslation();

  if (!counts || !queueStatus) {
    return null;
  }

  const { requested, predicted, max_queue_size } = counts;
  const { pending } = queueStatus;
  return (
    <Flex>
      <Tooltip
        label={
          requested > predicted &&
          t('queue.queueMaxExceeded', {
            requested,
            skip: requested - predicted,
            max_queue_size,
          })
        }
      >
        <Text
          variant="subtext"
          fontSize="sm"
          fontWeight={400}
          fontStyle="oblique 10deg"
          opacity={0.7}
          color={requested > predicted ? 'warning.500' : undefined}
        >
          {t('queue.queueCountPrediction', { predicted })}
        </Text>
      </Tooltip>
      <Spacer />
      <Text
        variant="subtext"
        fontSize="sm"
        fontWeight={400}
        fontStyle="oblique 10deg"
        opacity={0.7}
      >
        {pending > 0
          ? t('queue.queuePending', { pending })
          : t('queue.queueEmpty')}
      </Text>
    </Flex>
  );
});

QueueCounts.displayName = 'QueueCounts';
