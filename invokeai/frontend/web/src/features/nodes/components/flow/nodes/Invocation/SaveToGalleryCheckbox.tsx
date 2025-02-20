import { Checkbox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNodeHasImageOutput } from 'features/nodes/hooks/useNodeHasImageOutput';
import { useNodeIsIntermediate } from 'features/nodes/hooks/useNodeIsIntermediate';
import { nodeIsIntermediateChanged } from 'features/nodes/store/nodesSlice';
import { NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const SaveToGalleryCheckbox = ({ nodeId }: { nodeId: string }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const hasImageOutput = useNodeHasImageOutput(nodeId);
  const isIntermediate = useNodeIsIntermediate(nodeId);
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
    <FormControl className={NO_PAN_CLASS}>
      <FormLabel m={0}>{t('nodes.saveToGallery')} </FormLabel>
      <Checkbox onChange={handleChange} isChecked={!isIntermediate} />
    </FormControl>
  );
};

export default memo(SaveToGalleryCheckbox);
