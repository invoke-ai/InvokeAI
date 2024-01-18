import { Icon } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';

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
        <Icon as={PiWarningBold} color="error.300" />
      </InvTooltip>
    );
  }

  return null;
};

export default memo(StatusIndicator);
