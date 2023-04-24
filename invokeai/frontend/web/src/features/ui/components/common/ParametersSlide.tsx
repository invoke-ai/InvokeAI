import { Box, Flex, useOutsideClick } from '@chakra-ui/react';
import { Slide } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { isEqual } from 'lodash';
import { memo, PropsWithChildren, useRef } from 'react';
import PinParametersPanelButton from 'features/ui/components/PinParametersPanelButton';
import {
  setShouldShowParametersPanel,
  toggleParametersPanel,
  togglePinParametersPanel,
} from 'features/ui/store/uiSlice';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import Scrollable from 'features/ui/components/common/Scrollable';
import { useLangDirection } from 'features/ui/hooks/useDirection';
import { useHotkeys } from 'react-hotkeys-hook';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import AnimatedImageToImagePanel from 'features/parameters/components/AnimatedImageToImagePanel';

const parametersSlideSelector = createSelector(
  [uiSelector, generationSelector],
  (ui, generation) => {
    const { shouldPinParametersPanel, shouldShowParametersPanel } = ui;
    const { isImageToImageEnabled } = generation;

    return {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      isImageToImageEnabled,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type ParametersSlideProps = PropsWithChildren;

const ParametersSlide = (props: ParametersSlideProps) => {
  const dispatch = useAppDispatch();

  const {
    shouldShowParametersPanel,
    isImageToImageEnabled,
    shouldPinParametersPanel,
  } = useAppSelector(parametersSlideSelector);

  const langDirection = useLangDirection();

  const outsideClickRef = useRef<HTMLDivElement>(null);

  const closeParametersPanel = () => {
    dispatch(setShouldShowParametersPanel(false));
  };

  useOutsideClick({
    ref: outsideClickRef,
    handler: () => {
      closeParametersPanel();
    },
    enabled: shouldShowParametersPanel && !shouldPinParametersPanel,
  });

  useHotkeys(
    'o',
    () => {
      dispatch(toggleParametersPanel());
      shouldPinParametersPanel && dispatch(requestCanvasRescale());
    },
    [shouldPinParametersPanel]
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
      dispatch(togglePinParametersPanel());
      dispatch(requestCanvasRescale());
    },
    []
  );

  return (
    <Slide
      direction={langDirection === 'rtl' ? 'right' : 'left'}
      in={shouldShowParametersPanel}
      motionProps={{ initial: false }}
      style={{ zIndex: 99 }}
    >
      <Flex
        sx={{
          boxShadow: '0 0 4rem 0 rgba(0, 0, 0, 0.8)',
          pl: 4,
          py: 4,
          h: 'full',
          w: 'min',
          bg: 'base.900',
          borderInlineEndWidth: 4,
          borderInlineEndColor: 'base.800',
        }}
      >
        <Flex ref={outsideClickRef} position="relative" height="full" pr={4}>
          <Flex
            sx={{
              flexDirection: 'column',
              width: '28rem',
              flexShrink: 0,
            }}
          >
            <Flex
              paddingTop={1.5}
              paddingBottom={4}
              justifyContent="space-between"
              alignItems="center"
            >
              <InvokeAILogoComponent />
              <PinParametersPanelButton />
            </Flex>
            <Scrollable>{props.children}</Scrollable>
          </Flex>
          <AnimatedImageToImagePanel />
        </Flex>
      </Flex>
    </Slide>
  );
};

export default memo(ParametersSlide);
