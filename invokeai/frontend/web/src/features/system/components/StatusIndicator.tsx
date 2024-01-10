import { Icon } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaExclamationTriangle } from 'react-icons/fa';

const StatusIndicator = () => {
  const isConnected = useAppSelector((s) => s.system.isConnected);
  const { t } = useTranslation();

  if (!isConnected) {
    return (
      <InvTooltip
        label={t('common.statusDisconnected')}
        placement="end"
        shouldWrapChildren
        gutter={10}
      >
        <Icon as={FaExclamationTriangle} color="error.300" />
      </InvTooltip>
    );
  }

  return null;
};

export default memo(StatusIndicator);
