import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useBypassNode } from 'features/nodes/hooks/useBypassNode';
import { nodeBypassNodeChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const BypassNodeCheckbox = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const useBypass = useBypassNode(nodeId);
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        nodeBypassNodeChanged({
          nodeId,
          bypass: e.target.checked,
        })
      );
    },
    [dispatch, nodeId]
  );
  const { t } = useTranslation();

  return (
    <FormControl gap={2}>
      <Checkbox className="nopan" onChange={handleChange} isChecked={useBypass} />
      <FormLabel>{t('invocationCache.bypassNode')}</FormLabel>
    </FormControl>
  );
};

export default memo(BypassNodeCheckbox);
