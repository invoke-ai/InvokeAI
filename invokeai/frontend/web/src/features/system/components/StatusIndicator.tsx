import { Flex, Icon, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';
import { systemSelector } from '../store/systemSelectors';
import { ResourceKey } from 'i18next';
import { AnimatePresence, motion } from 'framer-motion';
import { useMemo, useRef } from 'react';
import { FaCircle } from 'react-icons/fa';
import { useHoverDirty } from 'react-use';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

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

  const statusColorScheme = useMemo(() => {
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
                color: `${statusColorScheme}.400`,
                pb: '1px',
                userSelect: 'none',
              }}
            >
              {t(statusTranslationKey as ResourceKey)}
              {iterationsText}
            </Text>
          </motion.div>
        )}
      </AnimatePresence>
      <Icon as={FaCircle} boxSize="0.5rem" color={`${statusColorScheme}.400`} />
    </Flex>
  );
};

export default StatusIndicator;
