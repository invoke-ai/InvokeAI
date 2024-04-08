import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { layerDeleted } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';

type Props = {
  id: string;
};

export const DeleteLayerButton = ({ id }: Props) => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(layerDeleted(id));
  }, [dispatch, id]);

  return <Button onClick={onClick} flexShrink={0}>Delete</Button>;
};
