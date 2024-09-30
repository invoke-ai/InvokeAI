import { MenuItem } from '@invoke-ai/ui-library';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { useImageActions } from 'features/gallery/hooks/useImageActions';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiAsteriskBold,
  PiPaintBrushBold,
  PiPlantBold,
  PiQuotesBold,
} from 'react-icons/pi';

export const ImageMenuItemMetadataRecallActions = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();

  const { recallAll, remix, recallSeed, recallPrompts, hasMetadata, hasSeed, hasPrompts, createAsPreset } =
    useImageActions(imageDTO);

  return (
    <>
      <MenuItem icon={<PiArrowsCounterClockwiseBold />} onClickCapture={remix} isDisabled={!hasMetadata}>
        {t('parameters.remixImage')}
      </MenuItem>
      <MenuItem icon={<PiQuotesBold />} onClickCapture={recallPrompts} isDisabled={!hasPrompts}>
        {t('parameters.usePrompt')}
      </MenuItem>
      <MenuItem icon={<PiPlantBold />} onClickCapture={recallSeed} isDisabled={!hasSeed}>
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem icon={<PiAsteriskBold />} onClickCapture={recallAll} isDisabled={!hasMetadata}>
        {t('parameters.useAll')}
      </MenuItem>
      <MenuItem icon={<PiPaintBrushBold />} onClickCapture={createAsPreset} isDisabled={!hasPrompts}>
        {t('stylePresets.useForTemplate')}
      </MenuItem>
    </>
  );
});

ImageMenuItemMetadataRecallActions.displayName = 'ImageMenuItemMetadataRecallActions';
