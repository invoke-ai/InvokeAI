import { Box, Flex } from '@chakra-ui/react';
import { PropsWithChildren, memo } from 'react';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import OverlayScrollable from './common/OverlayScrollable';
import PinParametersPanelButton from './PinParametersPanelButton';

type ParametersPinnedWrapperProps = PropsWithChildren;

const ParametersPinnedWrapper = (props: ParametersPinnedWrapperProps) => {
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
