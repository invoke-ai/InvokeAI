import { Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import React, { ReactNode, useCallback, useEffect, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { CSSTransition } from 'react-transition-group';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import {
  OptionsState,
  setOptionsPanelScrollPosition,
  setShouldHoldOptionsPanelOpen,
  setShouldPinOptionsPanel,
  setShouldShowOptionsPanel,
} from 'features/options/store/optionsSlice';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import InvokeAILogo from 'assets/images/logo.png';

type Props = { children: ReactNode };

const optionsPanelSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    const {
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
      shouldPinOptionsPanel,
      optionsPanelScrollPosition,
    } = options;

    return {
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
      shouldPinOptionsPanel,
      optionsPanelScrollPosition,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const InvokeOptionsPanel = (props: Props) => {
  const dispatch = useAppDispatch();
  const {
    shouldShowOptionsPanel,
    shouldHoldOptionsPanelOpen,
    shouldPinOptionsPanel,
  } = useAppSelector(optionsPanelSelector);

  const optionsPanelRef = useRef<HTMLDivElement>(null);
  const optionsPanelContainerRef = useRef<HTMLDivElement>(null);

  const timeoutIdRef = useRef<number | null>(null);

  const { children } = props;

  // Hotkeys
  useHotkeys(
    'o',
    () => {
      dispatch(setShouldShowOptionsPanel(!shouldShowOptionsPanel));
      shouldPinOptionsPanel &&
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowOptionsPanel, shouldPinOptionsPanel]
  );

  useHotkeys(
    'esc',
    () => {
      dispatch(setShouldShowOptionsPanel(false));
    },
    {
      enabled: () => !shouldPinOptionsPanel,
      preventDefault: true,
    },
    [shouldPinOptionsPanel]
  );

  useHotkeys(
    'shift+o',
    () => {
      handleClickPinOptionsPanel();
      dispatch(setDoesCanvasNeedScaling(true));
    },
    [shouldPinOptionsPanel]
  );

  const handleCloseOptionsPanel = useCallback(() => {
    if (shouldPinOptionsPanel) return;
    dispatch(
      setOptionsPanelScrollPosition(
        optionsPanelContainerRef.current
          ? optionsPanelContainerRef.current.scrollTop
          : 0
      )
    );
    dispatch(setShouldShowOptionsPanel(false));
    dispatch(setShouldHoldOptionsPanelOpen(false));
  }, [dispatch, shouldPinOptionsPanel]);

  const setCloseOptionsPanelTimer = () => {
    timeoutIdRef.current = window.setTimeout(
      () => handleCloseOptionsPanel(),
      500
    );
  };

  const cancelCloseOptionsPanelTimer = () => {
    timeoutIdRef.current && window.clearTimeout(timeoutIdRef.current);
  };

  const handleClickPinOptionsPanel = () => {
    dispatch(setShouldPinOptionsPanel(!shouldPinOptionsPanel));
    dispatch(setDoesCanvasNeedScaling(true));
  };

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        optionsPanelRef.current &&
        !optionsPanelRef.current.contains(e.target as Node)
      ) {
        handleCloseOptionsPanel();
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleCloseOptionsPanel]);

  return (
    <CSSTransition
      nodeRef={optionsPanelRef}
      in={
        shouldShowOptionsPanel ||
        (shouldHoldOptionsPanelOpen && !shouldPinOptionsPanel)
      }
      unmountOnExit
      timeout={200}
      classNames="options-panel-wrapper"
    >
      <div
        className="options-panel-wrapper"
        data-pinned={shouldPinOptionsPanel}
        tabIndex={1}
        ref={optionsPanelRef}
        onMouseEnter={
          !shouldPinOptionsPanel ? cancelCloseOptionsPanelTimer : undefined
        }
        onMouseOver={
          !shouldPinOptionsPanel ? cancelCloseOptionsPanelTimer : undefined
        }
        style={{
          borderRight: !shouldPinOptionsPanel
            ? '0.3rem solid var(--tab-list-text-inactive)'
            : '',
        }}
      >
        <div className="options-panel-margin">
          <div
            className="options-panel"
            ref={optionsPanelContainerRef}
            onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => {
              if (e.target !== optionsPanelContainerRef.current) {
                cancelCloseOptionsPanelTimer();
              } else {
                !shouldPinOptionsPanel && setCloseOptionsPanelTimer();
              }
            }}
          >
            <Tooltip label="Pin Options Panel">
              <div
                className="options-panel-pin-button"
                data-selected={shouldPinOptionsPanel}
                onClick={handleClickPinOptionsPanel}
              >
                {shouldPinOptionsPanel ? <BsPinAngleFill /> : <BsPinAngle />}
              </div>
            </Tooltip>
            {!shouldPinOptionsPanel && (
              <div className="invoke-ai-logo-wrapper">
                <img src={InvokeAILogo} alt="invoke-ai-logo" />
                <h1>
                  invoke <strong>ai</strong>
                </h1>
              </div>
            )}
            {children}
          </div>
        </div>
      </div>
    </CSSTransition>
  );
};

export default InvokeOptionsPanel;
