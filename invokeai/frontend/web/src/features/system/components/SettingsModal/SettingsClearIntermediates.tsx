import { Heading, Text } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import IAIButton from '../../../../common/components/IAIButton';
import {
  useClearIntermediatesMutation,
  useGetIntermediatesCountQuery,
} from '../../../../services/api/endpoints/images';
import { resetCanvas } from '../../../canvas/store/canvasSlice';
import { addToast } from '../../store/systemSlice';
import StyledFlex from './StyledFlex';

export default function SettingsClearIntermediates() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const { data: intermediatesCount } = useGetIntermediatesCountQuery(
    undefined,
    {
      refetchOnMountOrArgChange: true,
    }
  );

  const [clearIntermediates, { isLoading: isLoadingClearIntermediates }] =
    useClearIntermediatesMutation();

  const { data: queueStatus } = useGetQueueStatusQuery();
  const hasPendingItems =
    queueStatus &&
    (queueStatus.queue.in_progress > 0 || queueStatus.queue.pending > 0);

  const handleClickClearIntermediates = useCallback(() => {
    if (hasPendingItems) {
      return;
    }

    clearIntermediates()
      .unwrap()
      .then((clearedCount) => {
        dispatch(controlAdaptersReset());
        dispatch(resetCanvas());
        dispatch(
          addToast({
            title: t('settings.intermediatesCleared', { count: clearedCount }),
            status: 'info',
          })
        );
      })
      .catch(() => {
        dispatch(
          addToast({
            title: t('settings.intermediatesClearedFailed'),
            status: 'error',
          })
        );
      });
  }, [t, clearIntermediates, dispatch, hasPendingItems]);

  return (
    <StyledFlex>
      <Heading size="sm">{t('settings.clearIntermediates')}</Heading>
      <IAIButton
        tooltip={
          hasPendingItems ? t('settings.clearIntermediatesDisabled') : undefined
        }
        colorScheme="warning"
        onClick={handleClickClearIntermediates}
        isLoading={isLoadingClearIntermediates}
        isDisabled={!intermediatesCount || hasPendingItems}
      >
        {t('settings.clearIntermediatesWithCount', {
          count: intermediatesCount ?? 0,
        })}
      </IAIButton>
      <Text fontWeight="bold">{t('settings.clearIntermediatesDesc1')}</Text>
      <Text variant="subtext">{t('settings.clearIntermediatesDesc2')}</Text>
      <Text variant="subtext">{t('settings.clearIntermediatesDesc3')}</Text>
    </StyledFlex>
  );
}
