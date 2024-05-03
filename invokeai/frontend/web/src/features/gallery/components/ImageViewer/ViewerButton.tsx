import { Button } from '@invoke-ai/ui-library';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useImageViewer } from './useImageViewer';

export const ViewerButton = () => {
  const { t } = useTranslation();
  const { onOpen } = useImageViewer();
  const tooltip = useMemo(() => t('gallery.switchTo', { tab: t('common.viewer') }), [t]);

  return (
    <Button aria-label={tooltip} tooltip={tooltip} onClick={onOpen} variant="ghost" pointerEvents="auto">
      {t('common.viewer')}
    </Button>
  );
};
