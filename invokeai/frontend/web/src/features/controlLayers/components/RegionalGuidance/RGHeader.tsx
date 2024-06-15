import { Badge, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { RGActionsMenu } from 'features/controlLayers/components/RegionalGuidance/RGActionsMenu';
import { rgDeleted, rgIsEnabledToggled, selectRGOrThrow } from 'features/controlLayers/store/regionalGuidanceSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { RGMaskFillColorPicker } from './RGMaskFillColorPicker';
import { RGSettingsPopover } from './RGSettingsPopover';

type Props = {
  id: string;
  onToggleVisibility: () => void;
};

export const RGHeader = memo(({ id, onToggleVisibility }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).isEnabled);
  const autoNegative = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).autoNegative);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(rgIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(rgDeleted({ id }));
  }, [dispatch, id]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={t('controlLayers.regionalGuidance')} />
      <Spacer />
      {autoNegative === 'invert' && (
        <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
          {t('controlLayers.autoNegative')}
        </Badge>
      )}
      <RGMaskFillColorPicker id={id} />
      <RGSettingsPopover id={id} />
      <RGActionsMenu id={id} />
      <CanvasEntityDeleteButton onDelete={onDelete} />
    </CanvasEntityHeader>
  );
});

RGHeader.displayName = 'RGHeader';
