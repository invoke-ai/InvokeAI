import { Button } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowLeftBold } from 'react-icons/pi';

import { TAB_NAME_TO_TKEY, useImageViewer } from './useImageViewer';

export const BackToEditorButton = () => {
  const { t } = useTranslation();
  const { onClose } = useImageViewer();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const tooltip = useMemo(
    () => t('gallery.backToEditor', { tab: t(TAB_NAME_TO_TKEY[activeTabName]) }),
    [t, activeTabName]
  );

  return (
    <Button aria-label={tooltip} tooltip={tooltip} onClick={onClose} leftIcon={<PiArrowLeftBold />} variant="ghost">
      {t('common.back')}
    </Button>
  );
};
