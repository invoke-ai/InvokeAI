import { Flex, Icon } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvText } from 'common/components/InvText/wrapper';
import { STATUS_TRANSLATION_KEYS } from 'features/system/store/types';
import { AnimatePresence, motion } from 'framer-motion';
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
  ok: 'green.400',
  working: 'yellow.400',
  error: 'red.400',
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
    <Flex ref={ref} h="full" px={2} alignItems="center" gap={5}>
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
            <InvText
              sx={{
                fontSize: 'sm',
                fontWeight: 'semibold',
                pb: '1px',
                userSelect: 'none',
                color: COLOR_MAP[statusColor],
              }}
            >
              {t(statusTranslationKey as ResourceKey)}
            </InvText>
          </motion.div>
        )}
      </AnimatePresence>
      <Icon
        as={FaCircle}
        sx={{
          boxSize: '0.5rem',
          color: COLOR_MAP[statusColor],
        }}
      />
    </Flex>
  );
};

export default memo(StatusIndicator);
