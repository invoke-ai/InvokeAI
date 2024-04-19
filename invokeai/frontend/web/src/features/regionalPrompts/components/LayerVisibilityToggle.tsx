import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useLayerIsVisible } from 'features/regionalPrompts/hooks/layerStateHooks';
import { rpLayerIsVisibleToggled } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { PiEyeBold, PiEyeClosedBold } from 'react-icons/pi';

type Props = {
  id: string;
};

export const LayerVisibilityToggle = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isVisible = useLayerIsVisible(id);
  const onClick = useCallback(() => {
    dispatch(rpLayerIsVisibleToggled(id));
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
});

LayerVisibilityToggle.displayName = 'LayerVisibilityToggle';
