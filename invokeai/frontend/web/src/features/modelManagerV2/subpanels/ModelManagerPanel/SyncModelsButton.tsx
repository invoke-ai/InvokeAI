import { Button, Tooltip, useDisclosure } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsClockwiseBold } from 'react-icons/pi';

import { SyncModelsDialog } from './SyncModelsDialog';

export const SyncModelsButton = memo(() => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleClick = useCallback(() => {
    onOpen();
  }, [onOpen]);

  return (
    <>
      <Tooltip label={t('modelManager.syncModelsTooltip')}>
        <Button
          size="sm"
          colorScheme="error"
          leftIcon={<PiArrowsClockwiseBold />}
          onClick={handleClick}
          aria-label={t('modelManager.syncModels')}
        >
          {t('modelManager.syncModels')}
        </Button>
      </Tooltip>
      <SyncModelsDialog isOpen={isOpen} onClose={onClose} />
    </>
  );
});

SyncModelsButton.displayName = 'SyncModelsButton';
