import { IconMenuItem } from 'common/components/IconMenuItem';
import { openImageInNewTab } from 'common/util/openImageInNewTab';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ContextMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const onClick = useCallback(() => {
    openImageInNewTab(imageDTO.image_url);
  }, [imageDTO]);

  return (
    <IconMenuItem
      onClickCapture={onClick}
      aria-label={t('common.openInNewTab')}
      tooltip={t('common.openInNewTab')}
      icon={<PiArrowSquareOutBold />}
    />
  );
});

ContextMenuItemOpenInNewTab.displayName = 'ContextMenuItemOpenInNewTab';
