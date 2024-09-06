import { MenuItem } from '@invoke-ai/ui-library';
import { SpinnerIcon } from 'features/gallery/components/ImageContextMenu/SpinnerIcon';
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

  const {
    recallAll,
    remix,
    recallSeed,
    recallPrompts,
    hasMetadata,
    hasSeed,
    hasPrompts,
    isLoadingMetadata,
    createAsPreset,
  } = useImageActions(imageDTO?.image_name);

  return (
    <>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiArrowsCounterClockwiseBold />}
        onClickCapture={remix}
        isDisabled={isLoadingMetadata || !hasMetadata}
      >
        {t('parameters.remixImage')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiQuotesBold />}
        onClickCapture={recallPrompts}
        isDisabled={isLoadingMetadata || !hasPrompts}
      >
        {t('parameters.usePrompt')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiPlantBold />}
        onClickCapture={recallSeed}
        isDisabled={isLoadingMetadata || !hasSeed}
      >
        {t('parameters.useSeed')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiAsteriskBold />}
        onClickCapture={recallAll}
        isDisabled={isLoadingMetadata || !hasMetadata}
      >
        {t('parameters.useAll')}
      </MenuItem>
      <MenuItem
        icon={isLoadingMetadata ? <SpinnerIcon /> : <PiPaintBrushBold />}
        onClickCapture={createAsPreset}
        isDisabled={isLoadingMetadata || !hasPrompts}
      >
        {t('stylePresets.useForTemplate')}
      </MenuItem>
    </>
  );
});

ImageMenuItemMetadataRecallActions.displayName = 'ImageMenuItemMetadataRecallActions';
