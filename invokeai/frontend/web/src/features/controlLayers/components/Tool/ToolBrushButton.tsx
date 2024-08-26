import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { isDrawableEntityType } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiPaintBrushBold } from 'react-icons/pi';

export const ToolBrushButton = memo(() => {
  const { t } = useTranslation();
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector((s) => s.canvasSession.isStaging);
  const selectBrush = useSelectTool('brush');
  const isSelected = useToolIsSelected('brush');
  const isDrawingToolAllowed = useAppSelector((s) => {
    if (!s.canvasV2.selectedEntityIdentifier?.type) {
      return false;
    }
    return isDrawableEntityType(s.canvasV2.selectedEntityIdentifier.type);
  });

  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging || !isDrawingToolAllowed;
  }, [isDrawingToolAllowed, isFiltering, isStaging, isTransforming]);

  useHotkeys('b', selectBrush, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, selectBrush]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.brush')} (B)`}
      tooltip={`${t('controlLayers.tool.brush')} (B)`}
      icon={<PiPaintBrushBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={selectBrush}
      isDisabled={isDisabled}
    />
  );
});

ToolBrushButton.displayName = 'ToolBrushButton';
