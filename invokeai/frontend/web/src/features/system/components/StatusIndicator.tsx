import { Flex, Icon } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { STATUS_TRANSLATION_KEYS } from 'features/system/store/types';
import type { ResourceKey } from 'i18next';
import { memo, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCircle } from 'react-icons/fa';
import { useHoverDirty } from 'react-use';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const statusIndicatorSelector = createMemoizedSelector(
  stateSelector,
  ({ system }) => {
    const { isConnected, status } = system;

    return {
      isConnected,
      statusTranslationKey: STATUS_TRANSLATION_KEYS[status],
    };
  }
);

const COLOR_MAP = {
  ok: 'invokeYellow.500',
  working: 'blue.500',
  error: 'red.500',
};

const StatusIndicator = () => {
  const { isConnected, statusTranslationKey } = useAppSelector(
    statusIndicatorSelector
  );
  const { t } = useTranslation();
  const ref = useRef(null);
  const { data: queueStatus } = useGetQueueStatusQuery();

  const statusColor = useMemo(() => {
    if (!isConnected) {
      return 'error';
    }

    if (queueStatus?.queue.in_progress) {
      return 'working';
    }

    return 'ok';
  }, [queueStatus?.queue.in_progress, isConnected]);

  const isHovered = useHoverDirty(ref);

  return (
    <Flex ref={ref} alignItems="center" p={2} pl={0}>
      <InvTooltip
        left={10}
        bottom={-24}
        background="base.800"
        label={
          isHovered && (
            <InvText
              fontSize="sm"
              fontWeight="semibold"
              pb="1px"
              userSelect="none"
              color={COLOR_MAP[statusColor]}
            >
              {t(statusTranslationKey as ResourceKey)}
            </InvText>
          )
        }
      >
        <Icon as={FaCircle} boxSize="0.6rem" color={COLOR_MAP[statusColor]} />
      </InvTooltip>
    </Flex>
  );
};

export default memo(StatusIndicator);
