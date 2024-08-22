import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { isDrawableEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiPaintBrushBold } from 'react-icons/pi';

export const ToolBrushButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'brush');
  const isDrawingToolAllowed = useAppSelector((s) => {
    if (!s.canvasV2.selectedEntityIdentifier?.type) {
      return false;
    }
    return isDrawableEntityType(s.canvasV2.selectedEntityIdentifier.type);
  });

  const isDisabled = useMemo(() => {
    return isTransforming || isStaging || !isDrawingToolAllowed;
  }, [isDrawingToolAllowed, isStaging, isTransforming]);

  const onClick = useCallback(() => {
    dispatch(toolChanged('brush'));
  }, [dispatch]);

  useHotkeys('b', onClick, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, onClick]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.brush')} (B)`}
      tooltip={`${t('controlLayers.tool.brush')} (B)`}
      icon={<PiPaintBrushBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ToolBrushButton.displayName = 'ToolBrushButton';
