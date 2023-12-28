import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { sizeReset } from 'features/parameters/store/generationSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { IoSparkles } from 'react-icons/io5';

export const SetOptimalSizeButton = () => {
  const { t } = useTranslation();
  const optimalDimension = useAppSelector((state) =>
    state.generation.model?.base_model === 'sdxl' ? 1024 : 512
  );
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(sizeReset(optimalDimension));
  }, [dispatch, optimalDimension]);

  return (
    <InvIconButton
      aria-label={t('parameters.lockAspectRatio')}
      onClick={onClick}
      variant="ghost"
      size="sm"
      icon={<IoSparkles />}
    />
  );
};
