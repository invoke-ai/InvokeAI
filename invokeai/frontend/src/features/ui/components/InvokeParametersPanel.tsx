import { Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import {
  setShouldHoldParametersPanelOpen,
  setShouldPinParametersPanel,
  setShouldShowParametersPanel,
} from 'features/ui/store/uiSlice';

import React, { ReactNode, useCallback, useEffect, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { CSSTransition } from 'react-transition-group';

import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { setParametersPanelScrollPosition } from 'features/ui/store/uiSlice';

import InvokeAILogo from 'assets/images/logo.png';
import { isEqual } from 'lodash';
import { uiSelector } from '../store/uiSelectors';
import { useTranslation } from 'react-i18next';

type Props = { children: ReactNode };

const optionsPanelSelector = createSelector(
  uiSelector,
  (ui) => {
    const {
      shouldShowParametersPanel,
      shouldHoldParametersPanelOpen,
      shouldPinParametersPanel,
      parametersPanelScrollPosition,
    } = ui;

    return {
      shouldShowParametersPanel,
      shouldHoldParametersPanelOpen,
      shouldPinParametersPanel,
      parametersPanelScrollPosition,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const InvokeOptionsPanel = (props: Props) => {
  const dispatch = useAppDispatch();
  const {
    shouldShowParametersPanel,
    shouldHoldParametersPanelOpen,
    shouldPinParametersPanel,
  } = useAppSelector(optionsPanelSelector);

  const optionsPanelRef = useRef<HTMLDivElement>(null);
  const optionsPanelContainerRef = useRef<HTMLDivElement>(null);

  const timeoutIdRef = useRef<number | null>(null);

  const { children } = props;

  const { t } = useTranslation();

  // Hotkeys
  useHotkeys(
    'o',
    () => {
      dispatch(setShouldShowParametersPanel(!shouldShowParametersPanel));
      shouldPinParametersPanel &&
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowParametersPanel, shouldPinParametersPanel]
  );

  useHotkeys(
    'esc',
    () => {
      dispatch(setShouldShowParametersPanel(false));
    },
    {
      enabled: () => !shouldPinParametersPanel,
      preventDefault: true,
    },
    [shouldPinParametersPanel]
  );

  useHotkeys(
    'shift+o',
    () => {
      handleClickPinOptionsPanel();
      dispatch(setDoesCanvasNeedScaling(true));
    },
    [shouldPinParametersPanel]
  );

  const handleCloseOptionsPanel = useCallback(() => {
    if (shouldPinParametersPanel) return;
    dispatch(
      setParametersPanelScrollPosition(
        optionsPanelContainerRef.current
          ? optionsPanelContainerRef.current.scrollTop
          : 0
      )
    );
    dispatch(setShouldShowParametersPanel(false));
    dispatch(setShouldHoldParametersPanelOpen(false));
  }, [dispatch, shouldPinParametersPanel]);

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
    dispatch(setShouldPinParametersPanel(!shouldPinParametersPanel));
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
        shouldShowParametersPanel ||
        (shouldHoldParametersPanelOpen && !shouldPinParametersPanel)
      }
      unmountOnExit
      timeout={200}
      classNames="parameters-panel-wrapper"
    >
      <div
        className="parameters-panel-wrapper"
        data-pinned={shouldPinParametersPanel}
        tabIndex={1}
        ref={optionsPanelRef}
        onMouseEnter={
          !shouldPinParametersPanel ? cancelCloseOptionsPanelTimer : undefined
        }
        onMouseOver={
          !shouldPinParametersPanel ? cancelCloseOptionsPanelTimer : undefined
        }
        style={{
          borderRight: !shouldPinParametersPanel
            ? '0.3rem solid var(--tab-list-text-inactive)'
            : '',
        }}
      >
        <div className="parameters-panel-margin">
          <div
            className="parameters-panel"
            ref={optionsPanelContainerRef}
            onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => {
              if (e.target !== optionsPanelContainerRef.current) {
                cancelCloseOptionsPanelTimer();
              } else {
                !shouldPinParametersPanel && setCloseOptionsPanelTimer();
              }
            }}
          >
            <Tooltip label={t('common.pinOptionsPanel')}>
              <div
                className="parameters-panel-pin-button"
                data-selected={shouldPinParametersPanel}
                onClick={handleClickPinOptionsPanel}
              >
                {shouldPinParametersPanel ? <BsPinAngleFill /> : <BsPinAngle />}
              </div>
            </Tooltip>
            {!shouldPinParametersPanel && (
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
