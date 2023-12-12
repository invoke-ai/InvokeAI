import { Flex, Icon, Text } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { STATUS_TRANSLATION_KEYS } from 'features/system/store/types';
import { AnimatePresence, motion } from 'framer-motion';
import { ResourceKey } from 'i18next';
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

const DARK_COLOR_MAP = {
  ok: 'green.400',
  working: 'yellow.400',
  error: 'red.400',
};

const LIGHT_COLOR_MAP = {
  ok: 'green.600',
  working: 'yellow.500',
  error: 'red.500',
};

const StatusIcon = ({
  statusColor,
}: {
  statusColor: 'ok' | 'error' | 'working';
}) => {
  return (
    <Icon
      as={FaCircle}
      sx={{
        boxSize: '0.5rem',
        color: LIGHT_COLOR_MAP[statusColor],
        _dark: { color: DARK_COLOR_MAP[statusColor] },
      }}
    />
  );
};

const StatusText = ({
  statusColor,
}: {
  statusColor: 'ok' | 'error' | 'working';
}) => {
  const { t } = useTranslation();
  const { statusTranslationKey } = useAppSelector(statusIndicatorSelector);

  return (
    <Text
      sx={{
        fontSize: 'sm',
        fontWeight: '600',
        pb: '1px',
        userSelect: 'none',
        color: LIGHT_COLOR_MAP[statusColor],
        _dark: {
          color: DARK_COLOR_MAP[statusColor],
        },
      }}
    >
      {t(statusTranslationKey as ResourceKey)}
    </Text>
  );
};

type StatusIndicatorProps = {
  isSidePanelCollapsed?: boolean;
};

const StatusIndicator = (props: StatusIndicatorProps) => {
  const ref = useRef(null);
  const { isSidePanelCollapsed = false } = props;
  const { data: queueStatus } = useGetQueueStatusQuery();
  const { isConnected } = useAppSelector(statusIndicatorSelector);

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

  return !isSidePanelCollapsed ? (
    <Flex ref={ref} h="full" px={2} alignItems="center" gap={2}>
      <StatusIcon statusColor={statusColor} />
      <AnimatePresence>
        {isHovered && (
          <motion.div
            key="statusText"
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: { duration: 0.15 },
            }}
            exit={{
              opacity: 0,
              transition: { delay: 0.8 },
            }}
          >
            <StatusText statusColor={statusColor} />
          </motion.div>
        )}
      </AnimatePresence>
    </Flex>
  ) : (
    <StatusIcon statusColor={statusColor} />
  );
};

export default memo(StatusIndicator);
