import { Box, Flex, FlexProps, useColorMode } from '@chakra-ui/react';
import { memo } from 'react';
import { PanelResizeHandle } from 'react-resizable-panels';
import { mode } from 'theme/util/mode';

type ResizeHandleProps = FlexProps & {
  direction?: 'horizontal' | 'vertical';
};

const ResizeHandle = (props: ResizeHandleProps) => {
  const { direction = 'horizontal', ...rest } = props;
  const { colorMode } = useColorMode();

  if (direction === 'horizontal') {
    return (
      <PanelResizeHandle>
        <Flex
          sx={{
            w: 6,
            h: 'full',
            justifyContent: 'center',
            alignItems: 'center',
          }}
          {...rest}
        >
          <Box
            sx={{
              w: 0.5,
              h: 'calc(100% - 4px)',
              bg: mode('base.100', 'base.850')(colorMode),
            }}
          />
        </Flex>
      </PanelResizeHandle>
    );
  }

  return (
    <PanelResizeHandle>
      <Flex
        sx={{
          w: 'full',
          h: 6,
          justifyContent: 'center',
          alignItems: 'center',
        }}
        {...rest}
      >
        <Box
          sx={{
            w: 'calc(100% - 4px)',
            h: 0.5,
            bg: mode('base.100', 'base.850')(colorMode),
          }}
        />
      </Flex>
    </PanelResizeHandle>
  );
};

export default memo(ResizeHandle);
