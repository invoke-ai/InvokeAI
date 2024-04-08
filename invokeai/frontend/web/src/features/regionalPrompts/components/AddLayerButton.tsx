import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { layerAdded } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';

export const AddLayerButton = () => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(layerAdded('promptRegionLayer'));
  }, [dispatch]);

  return <Button onClick={onClick}>Add Layer</Button>;
};
