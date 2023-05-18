import { Box, Flex } from '@chakra-ui/react';
import { PropsWithChildren, memo } from 'react';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import OverlayScrollable from './common/OverlayScrollable';
import PinParametersPanelButton from './PinParametersPanelButton';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from '../store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';

const selector = createSelector(uiSelector, (ui) => {
  const { shouldPinParametersPanel, shouldShowParametersPanel } = ui;

  return {
    shouldPinParametersPanel,
    shouldShowParametersPanel,
  };
});

type ParametersPinnedWrapperProps = PropsWithChildren;

const ParametersPinnedWrapper = (props: ParametersPinnedWrapperProps) => {
  const { shouldPinParametersPanel, shouldShowParametersPanel } =
    useAppSelector(selector);

  if (!(shouldPinParametersPanel && shouldShowParametersPanel)) {
    return null;
  }

  return (
    <Box
      sx={{
        position: 'relative',
        h: 'full',
        w: PARAMETERS_PANEL_WIDTH,
        flexShrink: 0,
      }}
    >
      <OverlayScrollable>
        <Flex
          sx={{
            gap: 2,
            flexDirection: 'column',
            h: 'full',
            w: 'full',
            position: 'absolute',
          }}
        >
          {props.children}
        </Flex>
      </OverlayScrollable>
      <PinParametersPanelButton
        sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
      />
    </Box>
  );
};

export default memo(ParametersPinnedWrapper);
