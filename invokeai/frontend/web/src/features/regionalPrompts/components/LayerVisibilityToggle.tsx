import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useLayerIsVisible } from 'features/regionalPrompts/hooks/layerStateHooks';
import { layerIsVisibleToggled } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';
import { PiEyeBold, PiEyeClosedBold } from 'react-icons/pi';

type Props = {
  id: string;
};

export const LayerVisibilityToggle = ({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isVisible = useLayerIsVisible(id);
  const onClick = useCallback(() => {
    dispatch(layerIsVisibleToggled(id));
  }, [dispatch, id]);

  return (
    <IconButton
      size="sm"
      aria-label="Toggle layer visibility"
      variant={isVisible ? 'outline' : 'ghost'}
      icon={isVisible ? <PiEyeBold /> : <PiEyeClosedBold />}
      onClick={onClick}
    />
  );
};
