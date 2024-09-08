import { MenuItem } from '@invoke-ai/ui-library';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ImageMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();

  return (
    <MenuItem as="a" href={imageDTO.image_url} target="_blank" icon={<PiArrowSquareOutBold />}>
      {t('common.openInNewTab')}
    </MenuItem>
  );
});

ImageMenuItemOpenInNewTab.displayName = 'ImageMenuItemOpenInNewTab';
