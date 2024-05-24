import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useLayerIsEnabled } from 'features/controlLayers/hooks/layerStateHooks';
import { layerIsEnabledToggled } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';

type Props = {
  layerId: string;
};

export const LayerIsEnabledToggle = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useLayerIsEnabled(layerId);
  const onClick = useCallback(() => {
    dispatch(layerIsEnabledToggled(layerId));
  }, [dispatch, layerId]);

  return (
    <IconButton
      size="sm"
      aria-label={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      tooltip={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      variant="outline"
      icon={isEnabled ? <PiCheckBold /> : undefined}
      onClick={onClick}
      colorScheme="base"
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

LayerIsEnabledToggle.displayName = 'LayerVisibilityToggle';
