import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNonRasterLayersIsHidden } from 'features/controlLayers/hooks/useNonRasterLayersIsHidden';
import { allNonRasterLayersIsHiddenToggled } from 'features/controlLayers/store/canvasSlice';
import type { MouseEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeClosedBold } from 'react-icons/pi';

export const CanvasNonRasterLayersIsHiddenToggle = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isHidden = useNonRasterLayersIsHidden();
  
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

CanvasNonRasterLayersIsHiddenToggle.displayName = 'CanvasNonRasterLayersIsHiddenToggle';