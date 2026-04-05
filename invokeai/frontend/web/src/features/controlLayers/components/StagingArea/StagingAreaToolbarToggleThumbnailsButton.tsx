import { IconButton } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretLineDownBold, PiCaretLineUpBold } from 'react-icons/pi';

type Props = {
  areThumbnailsVisible: boolean;
  onToggle: () => void;
};

export const StagingAreaToolbarToggleThumbnailsButton = memo(({ areThumbnailsVisible, onToggle }: Props) => {
  const { t } = useTranslation();

  const label = areThumbnailsVisible
    ? t('controlLayers.stagingArea.hideThumbnails', { defaultValue: 'Hide Thumbnails' })
    : t('controlLayers.stagingArea.showThumbnails', { defaultValue: 'Show Thumbnails' });

  return (
    <IconButton
      tooltip={label}
      aria-label={label}
      icon={areThumbnailsVisible ? <PiCaretLineDownBold /> : <PiCaretLineUpBold />}
      onClick={onToggle}
      colorScheme="invokeBlue"
    />
  );
});

StagingAreaToolbarToggleThumbnailsButton.displayName = 'StagingAreaToolbarToggleThumbnailsButton';
