import { Button } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { layerReset } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';

type Props = {
  id: string;
};

export const ResetLayerButton = ({ id }: Props) => {
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(layerReset(id));
  }, [dispatch, id]);

  return <Button onClick={onClick} flexShrink={0}>Reset</Button>;
};
