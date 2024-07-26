import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { toolChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold } from 'react-icons/pi';

export const BboxToolButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isDisabled = useAppSelector((s) => s.canvasV2.session.isStaging || s.canvasV2.tool.isTransforming);
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'bbox');

  const onClick = useCallback(() => {
    dispatch(toolChanged('bbox'));
  }, [dispatch]);

  useHotkeys('q', onClick, { enabled: !isDisabled }, [onClick, isDisabled]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.bbox')} (Q)`}
      tooltip={`${t('controlLayers.bbox')} (Q)`}
      icon={<PiBoundingBoxBold />}
      colorScheme={isSelected ? 'invokeBlue' : 'base'}
      variant="outline"
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

BboxToolButton.displayName = 'BboxToolButton';
