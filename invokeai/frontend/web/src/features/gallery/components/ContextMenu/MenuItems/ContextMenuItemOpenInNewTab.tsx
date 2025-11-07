import { IconMenuItem } from 'common/components/IconMenuItem';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ContextMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const onClick = useCallback(() => {
    window.open(imageDTO.image_url, '_blank');
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
