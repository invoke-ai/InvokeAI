import { Heading } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvText } from 'common/components/InvText/wrapper';
import { resetCanvas } from 'features/canvas/store/canvasSlice';
import { controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import { addToast } from 'features/system/store/systemSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useClearIntermediatesMutation,
  useGetIntermediatesCountQuery,
} from 'services/api/endpoints/images';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

import StyledFlex from './StyledFlex';

const SettingsClearIntermediates = () => {
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
      <InvButton
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
      </InvButton>
      <InvText fontWeight="bold">
        {t('settings.clearIntermediatesDesc1')}
      </InvText>
      <InvText variant="subtext">
        {t('settings.clearIntermediatesDesc2')}
      </InvText>
      <InvText variant="subtext">
        {t('settings.clearIntermediatesDesc3')}
      </InvText>
    </StyledFlex>
  );
};

export default memo(SettingsClearIntermediates);
