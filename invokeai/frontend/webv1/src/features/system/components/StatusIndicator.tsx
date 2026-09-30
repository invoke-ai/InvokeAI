import { Icon, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';
import { $isConnected } from 'services/events/stores';

const StatusIndicator = () => {
  const isConnected = useStore($isConnected);
  const { t } = useTranslation();

  if (!isConnected) {
    return (
      <Tooltip label={t('common.statusDisconnected')} placement="end" shouldWrapChildren gutter={10}>
        <Icon as={PiWarningBold} color="error.300" />
      </Tooltip>
    );
  }

  return null;
};

export default memo(StatusIndicator);
