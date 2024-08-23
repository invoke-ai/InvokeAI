import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { isDrawableEntityType } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiCursorBold } from 'react-icons/pi';

export const ToolMoveButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isFiltering = useIsFiltering();
  const isTransforming = useIsTransforming();
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'move');
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
    dispatch(toolChanged('move'));
  }, [dispatch]);

  useHotkeys('v', onClick, { enabled: !isDisabled || isSelected }, [isDisabled, isSelected, onClick]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.move')} (V)`}
      tooltip={`${t('controlLayers.tool.move')} (V)`}
      icon={<PiCursorBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ToolMoveButton.displayName = 'ToolMoveButton';
