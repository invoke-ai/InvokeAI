import { Flex, Text, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  errorSeen,
  setShouldShowLogViewer,
  SystemState,
} from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { Resizable } from 're-resizable';
import { useLayoutEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaAngleDoubleDown, FaCode, FaMinus } from 'react-icons/fa';
import { systemSelector } from '../store/systemSelectors';

const logSelector = createSelector(
  systemSelector,
  (system: SystemState) => system.log,
  {
    memoizeOptions: {
      // We don't need a deep equality check for this selector.
      resultEqualityCheck: (a, b) => a.length === b.length,
    },
  }
);

const consoleSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    return {
      shouldShowLogViewer: system.shouldShowLogViewer,
      hasError: system.hasError,
      wasErrorSeen: system.wasErrorSeen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Basic log viewer, floats on bottom of page.
 */
const Console = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const log = useAppSelector(logSelector);
  const { shouldShowLogViewer, hasError, wasErrorSeen } =
    useAppSelector(consoleSelector);

  // Rudimentary autoscroll
  const [shouldAutoscroll, setShouldAutoscroll] = useState<boolean>(true);
  const viewerRef = useRef<HTMLDivElement>(null);

  /**
   * If autoscroll is on, scroll to the bottom when:
   * - log updates
   * - viewer is toggled
   *
   * Also scroll to the bottom whenever autoscroll is turned on.
   */
  useLayoutEffect(() => {
    if (viewerRef.current !== null && shouldAutoscroll) {
      viewerRef.current.scrollTop = viewerRef.current.scrollHeight;
    }
  }, [shouldAutoscroll, log, shouldShowLogViewer]);

  const handleClickLogViewerToggle = () => {
    dispatch(errorSeen());
    dispatch(setShouldShowLogViewer(!shouldShowLogViewer));
  };

  useHotkeys(
    '`',
    () => {
      dispatch(setShouldShowLogViewer(!shouldShowLogViewer));
    },
    [shouldShowLogViewer]
  );

  useHotkeys('esc', () => {
    dispatch(setShouldShowLogViewer(false));
  });

  const handleOnScroll = () => {
    if (!viewerRef.current) return;
    if (
      shouldAutoscroll &&
      viewerRef.current.scrollTop <
        viewerRef.current.scrollHeight - viewerRef.current.clientHeight
    ) {
      setShouldAutoscroll(false);
    }
  };

  return (
    <>
      {shouldShowLogViewer && (
        <Resizable
          defaultSize={{
            width: '100%',
            height: 200,
          }}
          style={{
            display: 'flex',
            position: 'fixed',
            insetInlineStart: 0,
            bottom: 0,
            zIndex: 9999,
          }}
          maxHeight="90vh"
        >
          <Flex
            sx={{
              flexDirection: 'column',
              width: '100vw',
              overflow: 'auto',
              direction: 'column',
              fontFamily: 'monospace',
              pt: 0,
              pr: 4,
              pb: 4,
              pl: 12,
              borderTopWidth: 5,
              bg: 'base.850',
              borderColor: 'base.700',
            }}
            ref={viewerRef}
            onScroll={handleOnScroll}
          >
            {log.map((entry, i) => {
              const { timestamp, message, level } = entry;
              const colorScheme = level === 'info' ? 'base' : level;
              return (
                <Flex
                  key={i}
                  sx={{
                    gap: 2,
                    color: `${colorScheme}.300`,
                  }}
                >
                  <Text fontWeight="600">{timestamp}:</Text>
                  <Text wordBreak="break-all">{message}</Text>
                </Flex>
              );
            })}
          </Flex>
        </Resizable>
      )}
      {shouldShowLogViewer && (
        <Tooltip
          hasArrow
          label={shouldAutoscroll ? 'Autoscroll On' : 'Autoscroll Off'}
        >
          <IAIIconButton
            size="sm"
            aria-label={t('accessibility.toggleAutoscroll')}
            icon={<FaAngleDoubleDown />}
            onClick={() => setShouldAutoscroll(!shouldAutoscroll)}
            isChecked={shouldAutoscroll}
            sx={{
              position: 'fixed',
              insetInlineStart: 2,
              bottom: 12,
              zIndex: '10000',
            }}
          />
        </Tooltip>
      )}
      <Tooltip
        hasArrow
        label={shouldShowLogViewer ? 'Hide Console' : 'Show Console'}
      >
        <IAIIconButton
          size="sm"
          aria-label={t('accessibility.toggleLogViewer')}
          icon={shouldShowLogViewer ? <FaMinus /> : <FaCode />}
          onClick={handleClickLogViewerToggle}
          sx={{
            position: 'fixed',
            insetInlineStart: 2,
            bottom: 2,
            zIndex: '10000',
          }}
          colorScheme={hasError || !wasErrorSeen ? 'error' : 'base'}
        />
      </Tooltip>
    </>
  );
};

export default Console;
