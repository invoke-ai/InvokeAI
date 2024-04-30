import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useLayerIsVisible } from 'features/controlLayers/hooks/layerStateHooks';
import { layerVisibilityToggled } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

type Props = {
  layerId: string;
};

export const RPLayerVisibilityToggle = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isVisible = useLayerIsVisible(layerId);
  const onClick = useCallback(() => {
    dispatch(layerVisibilityToggled(layerId));
  }, [dispatch, layerId]);

  return (
    <IconButton
      size="sm"
      aria-label={t('controlLayers.toggleVisibility')}
      tooltip={t('controlLayers.toggleVisibility')}
      variant="outline"
      icon={isVisible ? <PiCheckBold /> : undefined}
      onClick={onClick}
      colorScheme="base"
    />
  );
});

RPLayerVisibilityToggle.displayName = 'RPLayerVisibilityToggle';
