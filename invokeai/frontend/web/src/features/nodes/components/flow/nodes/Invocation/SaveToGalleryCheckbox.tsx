import { useAppDispatch } from 'app/store/storeHooks';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { useHasImageOutput } from 'features/nodes/hooks/useHasImageOutput';
import { useIsIntermediate } from 'features/nodes/hooks/useIsIntermediate';
import { nodeIsIntermediateChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const SaveToGalleryCheckbox = ({ nodeId }: { nodeId: string }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const hasImageOutput = useHasImageOutput(nodeId);
  const isIntermediate = useIsIntermediate(nodeId);
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        nodeIsIntermediateChanged({
          nodeId,
          isIntermediate: !e.target.checked,
        })
      );
    },
    [dispatch, nodeId]
  );

  if (!hasImageOutput) {
    return null;
  }

  return (
    <InvControl label={t('hotkeys.saveToGallery.title')} className="nopan">
      <InvCheckbox onChange={handleChange} isChecked={!isIntermediate} />
    </InvControl>
  );
};

export default memo(SaveToGalleryCheckbox);
