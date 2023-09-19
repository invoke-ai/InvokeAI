import { Flex, Icon, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { AnimatePresence, motion } from 'framer-motion';
import { ResourceKey } from 'i18next';
import { memo, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCircle } from 'react-icons/fa';
import { useHoverDirty } from 'react-use';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';
import { STATUS_TRANSLATION_KEYS } from '../store/types';

const statusIndicatorSelector = createSelector(
  stateSelector,
  ({ system }) => {
    const { isConnected, status } = system;

    return {
      isConnected,
      statusTranslationKey: STATUS_TRANSLATION_KEYS[status],
    };
  },
  defaultSelectorOptions
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

const StatusIndicator = () => {
  const { isConnected, statusTranslationKey } = useAppSelector(
    statusIndicatorSelector
  );
  const { t } = useTranslation();
  const ref = useRef(null);
  const { data: queueStatus } = useGetQueueStatusQuery();

  const statusColor = useMemo(() => {
    if (queueStatus?.queue.in_progress) {
      return 'working';
    }

    if (isConnected) {
      return 'ok';
    }

    return 'error';
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
            <Text
              sx={{
                fontSize: 'sm',
                fontWeight: '600',
                pb: '1px',
                userSelect: 'none',
                color: LIGHT_COLOR_MAP[statusColor],
                _dark: { color: DARK_COLOR_MAP[statusColor] },
              }}
            >
              {t(statusTranslationKey as ResourceKey)}
            </Text>
          </motion.div>
        )}
      </AnimatePresence>
      <Icon
        as={FaCircle}
        sx={{
          boxSize: '0.5rem',
          color: LIGHT_COLOR_MAP[statusColor],
          _dark: { color: DARK_COLOR_MAP[statusColor] },
        }}
      />
    </Flex>
  );
};

export default memo(StatusIndicator);
