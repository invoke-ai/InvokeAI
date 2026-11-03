import { IconMenuItem } from 'common/components/IconMenuItem';
import { openMediaInNewTab } from 'common/util/openMediaInNewTab';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const ContextMenuItemOpenInNewTab = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const onClick = useCallback(() => {
    openMediaInNewTab(imageDTO.image_url);
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
