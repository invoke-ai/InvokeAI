import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { iiIsEnabledToggled } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  onToggleVisibility: () => void;
};

export const InitialImageHeader = memo(({ onToggleVisibility }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => s.canvasV2.initialImage.isEnabled);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(iiIsEnabledToggled());
  }, [dispatch]);
  const title = useMemo(() => {
    return `${t('controlLayers.initialImage')}`;
  }, [t]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={title} />
      <Spacer />
    </CanvasEntityHeader>
  );
});

InitialImageHeader.displayName = 'InitialImageHeader';
