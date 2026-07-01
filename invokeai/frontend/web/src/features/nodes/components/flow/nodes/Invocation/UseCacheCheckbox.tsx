import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import { useUseCache } from 'features/nodes/hooks/useUseCache';
import { nodeUseCacheChanged } from 'features/nodes/store/nodesSlice';
import { NO_FIT_ON_DOUBLE_CLICK_CLASS, NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';

const UseCacheCheckbox = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const useCache = useUseCache();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();

  const isVisible = useMemo(() => {
    // In single-user mode (multiuser disabled), always show the checkbox
    if (setupStatus && !setupStatus.multiuser_enabled) {
      return true;
    }
    // In multiuser mode, only show the checkbox to admin users
    return currentUser?.is_admin ?? false;
  }, [setupStatus, currentUser]);

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        nodeUseCacheChanged({
          nodeId,
          useCache: e.target.checked,
        })
      );
    },
    [dispatch, nodeId]
  );
  const { t } = useTranslation();

  if (!isVisible) {
    return null;
  }

  return (
    <FormControl className={NO_FIT_ON_DOUBLE_CLICK_CLASS}>
      <FormLabel m={0}>{t('invocationCache.useCache')}</FormLabel>
      <Checkbox className={NO_PAN_CLASS} onChange={handleChange} isChecked={useCache} />
    </FormControl>
  );
};

export default memo(UseCacheCheckbox);
