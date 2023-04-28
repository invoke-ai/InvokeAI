import { Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { memo, ReactNode } from 'react';

import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import ResizableDrawer from 'features/ui/components/common/ResizableDrawer/ResizableDrawer';
import {
  setShouldShowParametersPanel,
  toggleParametersPanel,
  togglePinParametersPanel,
} from 'features/ui/store/uiSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import Scrollable from './common/Scrollable';
import PinParametersPanelButton from './PinParametersPanelButton';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector, uiSelector } from '../store/uiSelectors';
import { isEqual } from 'lodash-es';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import useResolution from 'common/hooks/useResolution';

const parametersPanelSelector = createSelector(
  [uiSelector, activeTabNameSelector, lightboxSelector],
  (ui, activeTabName, lightbox) => {
    const { shouldPinParametersPanel, shouldShowParametersPanel } = ui;
    const { isLightboxOpen } = lightbox;

    return {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      isResizable: activeTabName !== 'unifiedCanvas',
      isLightboxOpen,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type ParametersPanelProps = {
  children: ReactNode;
};

const ParametersPanel = ({ children }: ParametersPanelProps) => {
  const dispatch = useAppDispatch();

  const {
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    isResizable,
    isLightboxOpen,
  } = useAppSelector(parametersPanelSelector);

  const closeParametersPanel = () => {
    dispatch(setShouldShowParametersPanel(false));
  };

  const resolution = useResolution();

  useHotkeys(
    'o',
    () => {
      dispatch(toggleParametersPanel());
      shouldPinParametersPanel && dispatch(requestCanvasRescale());
    },
    { enabled: () => !isLightboxOpen },
    [shouldPinParametersPanel, isLightboxOpen]
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

  const parametersPanelContent = () => {
    return (
      <Flex
        flexDir="column"
        position="relative"
        h={{ base: 600, xl: 'full' }}
        w={{ sm: 'full', lg: '100vw', xl: 'full' }}
        paddingRight={{ base: 8, xl: 0 }}
      >
        {!shouldPinParametersPanel && (
          <Flex
            paddingTop={1.5}
            paddingBottom={4}
            justifyContent="space-between"
            alignItems="center"
          >
            <InvokeAILogoComponent />
            {resolution == 'desktop' && <PinParametersPanelButton />}
          </Flex>
        )}
        <Scrollable>{children}</Scrollable>
        {shouldPinParametersPanel && resolution == 'desktop' && (
          <PinParametersPanelButton
            sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
          />
        )}
      </Flex>
    );
  };

  const resizableParametersPanelContent = () => {
    return (
      <ResizableDrawer
        direction="left"
        isResizable={isResizable || !shouldPinParametersPanel}
        isOpen={shouldShowParametersPanel}
        onClose={closeParametersPanel}
        isPinned={shouldPinParametersPanel || isLightboxOpen}
        sx={{
          borderColor: 'base.700',
          p: shouldPinParametersPanel ? 0 : 4,
          bg: 'base.900',
        }}
        initialWidth={PARAMETERS_PANEL_WIDTH}
        minWidth={PARAMETERS_PANEL_WIDTH}
      >
        {parametersPanelContent()}
      </ResizableDrawer>
    );
  };

  const renderParametersPanel = () => {
    if (['mobile', 'tablet'].includes(resolution))
      return parametersPanelContent();
    return resizableParametersPanelContent();
  };

  return renderParametersPanel();
};

export default memo(ParametersPanel);
