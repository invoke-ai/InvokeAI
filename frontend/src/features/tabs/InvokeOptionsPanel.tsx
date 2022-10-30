import { createSelector } from '@reduxjs/toolkit';
import {
  FocusEvent,
  MouseEvent,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
} from 'react';
import { BsPinAngleFill } from 'react-icons/bs';
import { CSSTransition } from 'react-transition-group';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import useClickOutsideWatcher from '../../common/hooks/useClickOutsideWatcher';
import {
  OptionsState,
  setOptionsPanelScrollPosition,
  setShouldHoldOptionsPanelOpen,
  setShouldPinOptionsPanel,
  setShouldShowOptionsPanel,
} from '../options/optionsSlice';
import { setNeedsCache } from './Inpainting/inpaintingSlice';

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
  }
);

const InvokeOptionsPanel = (props: Props) => {
  const dispatch = useAppDispatch();
  const {
    shouldShowOptionsPanel,
    shouldHoldOptionsPanelOpen,
    shouldPinOptionsPanel,
    optionsPanelScrollPosition,
  } = useAppSelector(optionsPanelSelector);

  const optionsPanelRef = useRef<HTMLDivElement>(null);
  const optionsPanelContainerRef = useRef<HTMLDivElement>(null);

  const timeoutIdRef = useRef<number | null>(null);

  const { children } = props;

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
    // dispatch(setNeedsCache(true));
  }, [dispatch, shouldPinOptionsPanel]);

  useClickOutsideWatcher(optionsPanelRef, handleCloseOptionsPanel);

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
    dispatch(setNeedsCache(true));
  };

  // // set gallery scroll position
  // useEffect(() => {
  //   if (!optionsPanelContainerRef.current) return;
  //   optionsPanelContainerRef.current.scrollTop = optionsPanelScrollPosition;
  // }, [optionsPanelScrollPosition, shouldShowOptionsPanel]);

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
      >
        <div
          className="options-panel"
          ref={optionsPanelContainerRef}
          onMouseLeave={(e: MouseEvent<HTMLDivElement>) => {
            if (e.target !== optionsPanelContainerRef.current) {
              cancelCloseOptionsPanelTimer();
            } else {
              !shouldPinOptionsPanel && setCloseOptionsPanelTimer();
            }
          }}
        >
          <IAIIconButton
            size={'sm'}
            aria-label={'Pin Options Panel'}
            tooltip={'Pin Options Panel (Shift+P)'}
            onClick={handleClickPinOptionsPanel}
            icon={<BsPinAngleFill />}
            data-selected={shouldPinOptionsPanel}
          />
          {children}
        </div>
      </div>
    </CSSTransition>
  );
};

export default InvokeOptionsPanel;
