import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useIsTransforming } from 'features/controlLayers/hooks/useIsTransforming';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const ToolBboxButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isTransforming = useIsTransforming();
  const isStaging = useAppSelector((s) => s.canvasV2.session.isStaging);
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'bbox');
  const isDisabled = useMemo(() => {
    return isTransforming || isStaging;
  }, [isStaging, isTransforming]);

  const onClick = useCallback(() => {
    dispatch(toolChanged('bbox'));
  }, [dispatch]);

  useHotkeys('q', onClick, { enabled: !isDisabled || isSelected }, [onClick, isSelected, isDisabled]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.tool.bbox')} (Q)`}
      tooltip={`${t('controlLayers.tool.bbox')} (Q)`}
      icon={<PiBoundingBoxBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

ToolBboxButton.displayName = 'ToolBboxButton';
