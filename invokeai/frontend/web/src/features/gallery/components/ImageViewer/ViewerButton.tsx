import { IconButton } from '@chakra-ui/react';
import { Button } from '@invoke-ai/ui-library';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsDownUpBold } from 'react-icons/pi';

import { useImageViewer } from './useImageViewer';

export const ViewerButton = () => {
  const { t } = useTranslation();
  const { onOpen } = useImageViewer();

  const tooltip = useMemo(
    () =>
      t('gallery.switchTo', {
        tab: t('common.viewer'),
      }),
    [t]
  );

  return (
    <Button
      aria-label={tooltip}
      tooltip={tooltip}
      onClick={onOpen}
      variant="outline"
      pointerEvents="auto"
      sx={{ display: 'flex', gap: 2 }}
    >
      <IconButton aria-label={tooltip} variant="ghost" size="sm" icon={<PiArrowsDownUpBold />} />
      {t('common.viewer')}
    </Button>
  );
};
