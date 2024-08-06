import { Badge, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { RGActionsMenu } from 'features/controlLayers/components/RegionalGuidance/RGActionsMenu';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectRGOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { RGMaskFillColorPicker } from './RGMaskFillColorPicker';
import { RGSettingsPopover } from './RGSettingsPopover';

type Props = {
  onToggleVisibility: () => void;
};

export const RGHeader = memo(({ onToggleVisibility }: Props) => {
  const { id } = useEntityIdentifierContext();
  const { t } = useTranslation();
  const autoNegative = useAppSelector((s) => selectRGOrThrow(s.canvasV2, id).autoNegative);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle />
      <CanvasEntityTitle />
      <Spacer />
      {autoNegative === 'invert' && (
        <Badge color="base.300" bg="transparent" borderWidth={1} userSelect="none">
          {t('controlLayers.autoNegative')}
        </Badge>
      )}
      <RGMaskFillColorPicker />
      <RGSettingsPopover />
      <RGActionsMenu />
      <CanvasEntityDeleteButton />
    </CanvasEntityHeader>
  );
});

RGHeader.displayName = 'RGHeader';
