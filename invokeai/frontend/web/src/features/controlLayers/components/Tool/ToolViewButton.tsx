import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsFiltering } from 'features/controlLayers/hooks/useIsFiltering';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiHandBold } from 'react-icons/pi';

export const ToolViewButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isTransforming = useIsTransforming();
  const isFiltering = useIsFiltering();
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'view');
  const isDisabled = useMemo(() => {
    return isTransforming || isFiltering || isStaging;
  }, [isFiltering, isStaging, isTransforming]);
  const onClick = useCallback(() => {
    dispatch(toolChanged('view'));
  }, [dispatch]);

  useHotkeys('h', onClick, { enabled: !isDisabled || isSelected }, [onClick, isSelected, isDisabled]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.view')} (H)`}
      tooltip={`${t('controlLayers.tool.view')} (H)`}
      icon={<PiHandBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ToolViewButton.displayName = 'ToolViewButton';
