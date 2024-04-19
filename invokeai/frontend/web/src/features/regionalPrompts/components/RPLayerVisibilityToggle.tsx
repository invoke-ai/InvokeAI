import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useLayerIsVisible } from 'features/regionalPrompts/hooks/layerStateHooks';
import { rpLayerIsVisibleToggled } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeClosedBold } from 'react-icons/pi';

type Props = {
  layerId: string;
};

export const RPLayerVisibilityToggle = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isVisible = useLayerIsVisible(layerId);
  const onClick = useCallback(() => {
    dispatch(rpLayerIsVisibleToggled(layerId));
  }, [dispatch, layerId]);

  return (
    <IconButton
      size="sm"
      aria-label={t('regionalPrompts.toggleVisibility')}
      tooltip={t('regionalPrompts.toggleVisibility')}
      variant={isVisible ? 'outline' : 'ghost'}
      icon={isVisible ? <PiEyeBold /> : <PiEyeClosedBold />}
      onClick={onClick}
    />
  );
});

RPLayerVisibilityToggle.displayName = 'RPLayerVisibilityToggle';
