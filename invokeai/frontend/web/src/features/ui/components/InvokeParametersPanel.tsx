import { Box, Flex, Tooltip, Icon, useTheme } from '@chakra-ui/react';
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

import { isEqual } from 'lodash';
import { uiSelector } from '../store/uiSelectors';
import { useTranslation } from 'react-i18next';
import {
  APP_CONTENT_HEIGHT,
  OPTIONS_BAR_MAX_WIDTH,
  PROGRESS_BAR_THICKNESS,
} from 'theme/util/constants';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';

import './InvokeParametersPanel.css';
import { no_scrollbar } from 'theme/components/scrollbar';

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
  const { direction } = useTheme();

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
      classNames={`${direction}-parameters-panel-transition`}
    >
      <Box
        className={`${direction}-parameters-panel-transition`}
        tabIndex={1}
        ref={optionsPanelRef}
        onMouseEnter={
          !shouldPinParametersPanel ? cancelCloseOptionsPanelTimer : undefined
        }
        onMouseOver={
          !shouldPinParametersPanel ? cancelCloseOptionsPanelTimer : undefined
        }
        sx={{
          borderInlineEndWidth: !shouldPinParametersPanel ? 5 : 0,
          borderInlineEndStyle: 'solid',
          bg: 'base.900',
          borderColor: 'base.700',
          height: APP_CONTENT_HEIGHT,
          width: OPTIONS_BAR_MAX_WIDTH,
          maxWidth: OPTIONS_BAR_MAX_WIDTH,
          flexShrink: 0,
          position: 'relative',
          overflowY: 'scroll',
          overflowX: 'hidden',
          ...no_scrollbar,
          ...(!shouldPinParametersPanel && {
            zIndex: 20,
            position: 'fixed',
            top: 0,
            insetInlineStart: 0,
            width: `calc(${OPTIONS_BAR_MAX_WIDTH} + 2rem)`,
            maxWidth: `calc(${OPTIONS_BAR_MAX_WIDTH} + 2rem)`,
            height: '100%',
          }),
        }}
      >
        <Box sx={{ margin: !shouldPinParametersPanel && 4 }}>
          <Flex
            ref={optionsPanelContainerRef}
            onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => {
              if (e.target !== optionsPanelContainerRef.current) {
                cancelCloseOptionsPanelTimer();
              } else {
                !shouldPinParametersPanel && setCloseOptionsPanelTimer();
              }
            }}
            sx={{
              display: 'flex',
              flexDirection: 'column',
              rowGap: 2,
              height: '100%',
            }}
          >
            <Tooltip label={t('common.pinOptionsPanel')}>
              <Box
                onClick={handleClickPinOptionsPanel}
                sx={{
                  position: 'absolute',
                  cursor: 'pointer',
                  padding: 2,
                  top: 4,
                  insetInlineEnd: 4,
                  zIndex: 20,
                  ...(shouldPinParametersPanel && {
                    top: 0,
                    insetInlineEnd: 0,
                  }),
                }}
              >
                <Icon
                  sx={{ opacity: 0.2 }}
                  as={shouldPinParametersPanel ? BsPinAngleFill : BsPinAngle}
                />
              </Box>
            </Tooltip>
            {!shouldPinParametersPanel && (
              <Box sx={{ pt: PROGRESS_BAR_THICKNESS, pb: 2 }}>
                <InvokeAILogoComponent />
              </Box>
            )}
            {children}
          </Flex>
        </Box>
      </Box>
    </CSSTransition>
  );
};

export default InvokeOptionsPanel;
