import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { IMActionsMenu } from 'features/controlLayers/components/InpaintMask/IMActionsMenu';
import { imIsEnabledToggled } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { IMMaskFillColorPicker } from './IMMaskFillColorPicker';

type Props = {
  onToggleVisibility: () => void;
};

export const IMHeader = memo(({ onToggleVisibility }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => s.canvasV2.inpaintMask.isEnabled);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(imIsEnabledToggled());
  }, [dispatch]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={t('controlLayers.inpaintMask')} />
      <Spacer />
      <IMMaskFillColorPicker />
      <IMActionsMenu />
    </CanvasEntityHeader>
  );
});

IMHeader.displayName = 'IMHeader';
