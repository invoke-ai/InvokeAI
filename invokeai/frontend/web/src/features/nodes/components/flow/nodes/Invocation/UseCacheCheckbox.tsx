import { useAppDispatch } from 'app/store/storeHooks';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { useUseCache } from 'features/nodes/hooks/useUseCache';
import { nodeUseCacheChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const UseCacheCheckbox = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const useCache = useUseCache(nodeId);
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
  return (
    <InvControl label={t('invocationCache.useCache')}>
      <InvCheckbox
        className="nopan"
        onChange={handleChange}
        isChecked={useCache}
      />
    </InvControl>
  );
};

export default memo(UseCacheCheckbox);
