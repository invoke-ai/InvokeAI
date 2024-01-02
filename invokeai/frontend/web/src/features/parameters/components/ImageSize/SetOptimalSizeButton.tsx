import { useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { useImageSizeContext } from 'features/parameters/components/ImageSize/ImageSizeContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoSparkles } from 'react-icons/io5';

export const SetOptimalSizeButton = memo(() => {
  const { t } = useTranslation();
  const ctx = useImageSizeContext();
  const optimalDimension = useAppSelector((state) =>
    state.generation.model?.base_model === 'sdxl' ? 1024 : 512
  );
  const onClick = useCallback(() => {
    ctx.sizeReset(optimalDimension, optimalDimension);
  }, [ctx, optimalDimension]);

  return (
    <InvIconButton
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<IoSparkles />}
    />
  );
});

SetOptimalSizeButton.displayName = 'SetOptimalSizeButton';
