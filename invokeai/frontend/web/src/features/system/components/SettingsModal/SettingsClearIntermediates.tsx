import { Heading, Text } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import { useCallback, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import IAIButton from '../../../../common/components/IAIButton';
import {
  useClearIntermediatesMutation,
  useGetIntermediatesCountQuery,
} from '../../../../services/api/endpoints/images';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { resetCanvas } from '../../../canvas/store/canvasSlice';
import { addToast } from '../../store/systemSlice';
import StyledFlex from './StyledFlex';

export default function SettingsClearIntermediates() {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const { data: intermediatesCount, refetch: updateIntermediatesCount } =
    useGetIntermediatesCountQuery();

  const [clearIntermediates, { isLoading: isLoadingClearIntermediates }] =
    useClearIntermediatesMutation();

  const { data: queueStatus } = useGetQueueStatusQuery();
  const hasPendingItems = queueStatus && (queueStatus.queue.in_progress > 0 || queueStatus.queue.pending > 0);

  const handleClickClearIntermediates = useCallback(() => {
    if (hasPendingItems) {
      return;
    }

    clearIntermediates()
      .unwrap()
      .then((number) => {
        dispatch(controlAdaptersReset());
        dispatch(resetCanvas());
        dispatch(
          addToast({
            title:
              number === 1
                ? t('settings.intermediatesCleared_one')
                : t('settings.intermediatesCleared_other', { number }),
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

  useEffect(() => {
    // update the count on mount
    updateIntermediatesCount();
  }, [updateIntermediatesCount]);

  const buttonText = useMemo(() => {
    if (!intermediatesCount) {
      return t('settings.noIntermediates');
    }
    if (intermediatesCount === 1) {
      return t('settings.clearIntermediates_one');
    }
    return t('settings.clearIntermediates_other', {
      number: intermediatesCount,
    });
  }, [intermediatesCount, t]);

  return (
    <StyledFlex>
      <Heading size="sm">{t('settings.clearIntermediates')}</Heading>
      <IAIButton
        colorScheme="warning"
        onClick={handleClickClearIntermediates}
        isLoading={isLoadingClearIntermediates}
        isDisabled={!intermediatesCount}
      >
        {buttonText}
      </IAIButton>
      <Text fontWeight="bold">{t('settings.clearIntermediatesDesc1')}</Text>
      <Text variant="subtext">{t('settings.clearIntermediatesDesc2')}</Text>
      <Text variant="subtext">{t('settings.clearIntermediatesDesc3')}</Text>
    </StyledFlex>
  );
}
