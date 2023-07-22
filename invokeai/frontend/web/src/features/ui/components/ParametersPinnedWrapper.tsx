import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { PropsWithChildren, memo } from 'react';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import { uiSelector } from '../store/uiSelectors';
import PinParametersPanelButton from './PinParametersPanelButton';

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
      <Flex
        sx={{
          gap: 2,
          flexDirection: 'column',
          h: 'full',
          w: 'full',
          position: 'absolute',
          overflowY: 'auto',
        }}
      >
        {props.children}
      </Flex>

      <PinParametersPanelButton
        sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
      />
    </Box>
  );
};

export default memo(ParametersPinnedWrapper);
