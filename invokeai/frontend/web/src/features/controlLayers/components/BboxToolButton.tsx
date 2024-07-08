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
  const isDisabled = useAppSelector((s) => s.canvasV2.session.isStaging);
  const isSelected = useAppSelector((s) => s.canvasV2.tool.selected === 'bbox');

  const onClick = useCallback(() => {
    dispatch(toolChanged('bbox'));
  }, [dispatch]);

  useHotkeys('q', onClick, [onClick]);

  return (
    <IconButton
      aria-label={`${t('controlLayers.bbox')} (Q)`}
      tooltip={`${t('controlLayers.bbox')} (Q)`}
      icon={<PiBoundingBoxBold />}
      variant={isSelected ? 'solid' : 'outline'}
      onClick={onClick}
      isDisabled={isDisabled}
    />
  );
});

BboxToolButton.displayName = 'BboxToolButton';
