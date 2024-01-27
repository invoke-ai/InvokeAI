import type { ChakraProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { useCancelCurrentQueueItem } from 'features/queue/hooks/useCancelCurrentQueueItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
type Props = {
  sx?: ChakraProps['sx'];
};

const CancelCurrentQueueItemIconButton = ({ sx }: Props) => {
  const { t } = useTranslation();
  const { cancelQueueItem, isLoading, isDisabled } = useCancelCurrentQueueItem();

  return (
    <IconButton
      isDisabled={isDisabled}
      isLoading={isLoading}
      aria-label={t('queue.cancel')}
      tooltip={t('queue.cancelTooltip')}
      icon={<PiXBold size="16px" />}
      onClick={cancelQueueItem}
      colorScheme="error"
      sx={sx}
    />
  );
};

export default memo(CancelCurrentQueueItemIconButton);
