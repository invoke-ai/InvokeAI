import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircleFill } from 'react-icons/pi';

export const AutoAddIndicator = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex color="invokeBlue.300" alignItems="center">
      <PiCircleFill aria-label={t('common.auto')} size={10} />
    </Flex>
  );
});

AutoAddIndicator.displayName = 'AutoAddIndicator';
