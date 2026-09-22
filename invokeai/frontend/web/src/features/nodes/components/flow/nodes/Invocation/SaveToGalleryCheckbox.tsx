import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNodeHasGalleryOutput } from 'features/nodes/hooks/useNodeHasGalleryOutput';
import { useNodeIsIntermediate } from 'features/nodes/hooks/useNodeIsIntermediate';
import { nodeIsIntermediateChanged } from 'features/nodes/store/nodesSlice';
import { NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const SaveToGalleryCheckbox = ({ nodeId }: { nodeId: string }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const hasGalleryOutput = useNodeHasGalleryOutput();
  const isIntermediate = useNodeIsIntermediate();
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

  if (!hasGalleryOutput) {
    return null;
  }

  return (
    <FormControl className={NO_PAN_CLASS}>
      <FormLabel m={0}>{t('nodes.saveToGallery')} </FormLabel>
      <Checkbox onChange={handleChange} isChecked={!isIntermediate} />
    </FormControl>
  );
};

export default memo(SaveToGalleryCheckbox);
