import { ButtonGroup } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetInvocationCacheStatusQuery } from 'services/api/endpoints/appInfo';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import ClearInvocationCacheButton from './ClearInvocationCacheButton';
import ToggleInvocationCacheButton from './ToggleInvocationCacheButton';
import StatusStatGroup from './common/StatusStatGroup';
import StatusStatItem from './common/StatusStatItem';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';

const InvocationCacheStatus = () => {
  const { t } = useTranslation();
  const isConnected = useAppSelector((state) => state.system.isConnected);
  const isInvocationCacheEnabled =
    useFeatureStatus('invocationCache').isFeatureEnabled;
  const { data: queueStatus } = useGetQueueStatusQuery(undefined);
  const { data: cacheStatus } = useGetInvocationCacheStatusQuery(undefined, {
    pollingInterval:
      isConnected &&
      queueStatus?.processor.is_started &&
      queueStatus?.queue.pending > 0
        ? 5000
        : 0,
  });

  return (
    <StatusStatGroup>
      <StatusStatItem
        isDisabled={!cacheStatus?.enabled}
        label={t('invocationCache.cacheSize')}
        value={cacheStatus?.size ?? 0}
      />
      <StatusStatItem
        isDisabled={!cacheStatus?.enabled}
        label={t('invocationCache.hits')}
        value={cacheStatus?.hits ?? 0}
      />
      <StatusStatItem
        isDisabled={!cacheStatus?.enabled}
        label={t('invocationCache.misses')}
        value={cacheStatus?.misses ?? 0}
      />
      <StatusStatItem
        isDisabled={!cacheStatus?.enabled}
        label={t('invocationCache.maxCacheSize')}
        value={cacheStatus?.max_size ?? 0}
      />
      <ButtonGroup w={24} orientation="vertical" size="xs">
        <ClearInvocationCacheButton />
        <ToggleInvocationCacheButton />
      </ButtonGroup>
    </StatusStatGroup>
  );
};

export default memo(InvocationCacheStatus);
