import { IconButton, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import {
  errorSeen,
  setShouldShowLogViewer,
  SystemState,
} from 'features/system/store/systemSlice';
import { isEqual } from 'lodash';
import { Resizable } from 're-resizable';
import { useLayoutEffect, useRef, useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
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
            left: 0,
            bottom: 0,
            zIndex: 9999,
          }}
          maxHeight="90vh"
        >
          <div className="console" ref={viewerRef} onScroll={handleOnScroll}>
            {log.map((entry, i) => {
              const { timestamp, message, level } = entry;
              return (
                <div key={i} className={`console-entry console-${level}-color`}>
                  <p className="console-timestamp">{timestamp}:</p>
                  <p className="console-message">{message}</p>
                </div>
              );
            })}
          </div>
        </Resizable>
      )}
      {shouldShowLogViewer && (
        <Tooltip
          hasArrow
          label={shouldAutoscroll ? 'Autoscroll On' : 'Autoscroll Off'}
        >
          <IconButton
            className="console-autoscroll-icon-button"
            data-autoscroll-enabled={shouldAutoscroll}
            size="sm"
            aria-label="Toggle autoscroll"
            variant="solid"
            icon={<FaAngleDoubleDown />}
            onClick={() => setShouldAutoscroll(!shouldAutoscroll)}
          />
        </Tooltip>
      )}
      <Tooltip
        hasArrow
        label={shouldShowLogViewer ? 'Hide Console' : 'Show Console'}
      >
        <IconButton
          className="console-toggle-icon-button"
          data-error-seen={hasError || !wasErrorSeen}
          size="sm"
          position="fixed"
          variant="solid"
          aria-label="Toggle Log Viewer"
          icon={shouldShowLogViewer ? <FaMinus /> : <FaCode />}
          onClick={handleClickLogViewerToggle}
        />
      </Tooltip>
    </>
  );
};

export default Console;
