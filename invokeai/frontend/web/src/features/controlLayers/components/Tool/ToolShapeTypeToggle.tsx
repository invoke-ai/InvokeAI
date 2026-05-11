import { ButtonGroup, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectShapeType, settingsShapeTypeChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircleBold, PiPolygonBold, PiRectangleBold, PiScribbleLoopBold } from 'react-icons/pi';

export const ToolShapeTypeToggle = memo(() => {
  const { t } = useTranslation();
  const shapeType = useAppSelector(selectShapeType);
  const dispatch = useAppDispatch();

  const onRectClick = useCallback(() => dispatch(settingsShapeTypeChanged('rect')), [dispatch]);
  const onOvalClick = useCallback(() => dispatch(settingsShapeTypeChanged('oval')), [dispatch]);
  const onPolygonClick = useCallback(() => dispatch(settingsShapeTypeChanged('polygon')), [dispatch]);
  const onFreehandClick = useCallback(() => dispatch(settingsShapeTypeChanged('freehand')), [dispatch]);

  const rectLabel = t('controlLayers.shape.rect', { defaultValue: 'Rect' });
  const ovalLabel = t('controlLayers.shape.oval', { defaultValue: 'Oval' });
  const polygonLabel = t('controlLayers.lasso.polygon', { defaultValue: 'Polygon' });
  const freehandLabel = t('controlLayers.lasso.freehand', { defaultValue: 'Freehand' });

  return (
    <ButtonGroup isAttached size="sm">
      <Tooltip label={rectLabel}>
        <IconButton
          aria-label={rectLabel}
          icon={<PiRectangleBold />}
          colorScheme={shapeType === 'rect' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={onRectClick}
        />
      </Tooltip>
      <Tooltip label={ovalLabel}>
        <IconButton
          aria-label={ovalLabel}
          icon={<PiCircleBold />}
          colorScheme={shapeType === 'oval' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={onOvalClick}
        />
      </Tooltip>
      <Tooltip label={polygonLabel}>
        <IconButton
          aria-label={polygonLabel}
          icon={<PiPolygonBold />}
          colorScheme={shapeType === 'polygon' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={onPolygonClick}
        />
      </Tooltip>
      <Tooltip label={freehandLabel}>
        <IconButton
          aria-label={freehandLabel}
          icon={<PiScribbleLoopBold />}
          colorScheme={shapeType === 'freehand' ? 'invokeBlue' : 'base'}
          variant="solid"
          onClick={onFreehandClick}
        />
      </Tooltip>
    </ButtonGroup>
  );
});

ToolShapeTypeToggle.displayName = 'ToolShapeTypeToggle';
