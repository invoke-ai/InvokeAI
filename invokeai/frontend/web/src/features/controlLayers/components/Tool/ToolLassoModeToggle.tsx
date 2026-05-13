import { ButtonGroup, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectLassoMode, settingsLassoModeChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPolygonBold, PiScribbleLoopBold } from 'react-icons/pi';

export const ToolLassoModeToggle = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const lassoMode = useAppSelector(selectLassoMode);

  const setFreehand = useCallback(() => {
    dispatch(settingsLassoModeChanged('freehand'));
  }, [dispatch]);

  const setPolygon = useCallback(() => {
    dispatch(settingsLassoModeChanged('polygon'));
  }, [dispatch]);

  return (
    <ButtonGroup isAttached size="sm">
      <Tooltip label={t('controlLayers.lasso.freehand', { defaultValue: 'Freehand' })}>
        <IconButton
          aria-label={t('controlLayers.lasso.freehand', { defaultValue: 'Freehand' })}
          icon={<PiScribbleLoopBold size={16} />}
          colorScheme={lassoMode === 'freehand' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={setFreehand}
        />
      </Tooltip>
      <Tooltip label={t('controlLayers.lasso.polygon', { defaultValue: 'Polygon' })}>
        <IconButton
          aria-label={t('controlLayers.lasso.polygonHint', {
            defaultValue: 'Click to add points, click the first point to close.',
          })}
          icon={<PiPolygonBold size={16} />}
          colorScheme={lassoMode === 'polygon' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={setPolygon}
        />
      </Tooltip>
    </ButtonGroup>
  );
});

ToolLassoModeToggle.displayName = 'ToolLassoModeToggle';
