import { Flex, Icon, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { AnimatePresence, motion } from 'framer-motion';
import { ResourceKey } from 'i18next';
import { memo, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCircle } from 'react-icons/fa';
import { useHoverDirty } from 'react-use';
import { systemSelector } from '../store/systemSelectors';

const statusIndicatorSelector = createSelector(
  systemSelector,
  (system) => {
    const {
      isConnected,
      isProcessing,
      statusTranslationKey,
      currentIteration,
      totalIterations,
      currentStatusHasSteps,
    } = system;

    return {
      isConnected,
      isProcessing,
      currentIteration,
      totalIterations,
      statusTranslationKey,
      currentStatusHasSteps,
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
  const {
    isConnected,
    isProcessing,
    currentIteration,
    totalIterations,
    statusTranslationKey,
  } = useAppSelector(statusIndicatorSelector);
  const { t } = useTranslation();
  const ref = useRef(null);

  const statusString = useMemo(() => {
    if (isProcessing) {
      return 'working';
    }

    if (isConnected) {
      return 'ok';
    }

    return 'error';
  }, [isProcessing, isConnected]);

  const iterationsText = useMemo(() => {
    if (!(currentIteration && totalIterations)) {
      return;
    }

    return ` (${currentIteration}/${totalIterations})`;
  }, [currentIteration, totalIterations]);

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
                color: LIGHT_COLOR_MAP[statusString],
                _dark: { color: DARK_COLOR_MAP[statusString] },
              }}
            >
              {t(statusTranslationKey as ResourceKey)}
              {iterationsText}
            </Text>
          </motion.div>
        )}
      </AnimatePresence>
      <Icon
        as={FaCircle}
        sx={{
          boxSize: '0.5rem',
          color: LIGHT_COLOR_MAP[statusString],
          _dark: { color: DARK_COLOR_MAP[statusString] },
        }}
      />
    </Flex>
  );
};

export default memo(StatusIndicator);
