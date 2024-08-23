import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { isDrawableEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiRectangleBold } from 'react-icons/pi';

export const ToolRectButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'rect');
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);
  const isDrawingToolAllowed = useAppSelector((s) => {
    if (!s.canvasV2.selectedEntityIdentifier?.type) {
      return false;
    }
    return isDrawableEntityType(s.canvasV2.selectedEntityIdentifier.type);
  });

  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging || !isDrawingToolAllowed;
  }, [isDrawingToolAllowed, isFiltering, isStaging, isTransforming]);

  const onClick = useCallback(() => {
    dispatch(toolChanged('rect'));
  }, [dispatch]);

  useHotkeys('u', onClick, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, onClick]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.rectangle')} (U)`}
      tooltip={`${t('controlLayers.tool.rectangle')} (U)`}
      icon={<PiRectangleBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ToolRectButton.displayName = 'ToolRectButton';
