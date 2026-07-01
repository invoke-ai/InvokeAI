import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { allNonRasterLayersIsHiddenToggled } from 'features/controlLayers/store/canvasSlice';
import { selectNonRasterLayersIsHidden } from 'features/controlLayers/store/selectors';
import type { MouseEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeClosedBold } from 'react-icons/pi';

export const EntityListNonRasterLayerToggle = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isHidden = useAppSelector(selectNonRasterLayersIsHidden);

  const onClick = useCallback<MouseEventHandler>(
    (e) => {
      e.stopPropagation();
      dispatch(allNonRasterLayersIsHiddenToggled());
    },
    [dispatch]
  );

  return (
    <IconButton
      size="sm"
      aria-label={t(isHidden ? 'controlLayers.showNonRasterLayers' : 'controlLayers.hideNonRasterLayers')}
      tooltip={t(isHidden ? 'controlLayers.showNonRasterLayers' : 'controlLayers.hideNonRasterLayers')}
      variant="link"
      icon={isHidden ? <PiEyeClosedBold /> : <PiEyeBold />}
      onClick={onClick}
      alignSelf="stretch"
    />
  );
});

EntityListNonRasterLayerToggle.displayName = 'EntityListNonRasterLayerToggle';
