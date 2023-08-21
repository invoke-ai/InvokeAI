import { Box, Flex } from '@chakra-ui/react';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { PropsWithChildren, memo } from 'react';

const ScrollableContent = (props: PropsWithChildren) => {
  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        position: 'relative',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
        }}
      >
        <OverlayScrollbarsComponent
          defer
          style={{ height: '100%', width: '100%' }}
          options={{
            scrollbars: {
              visibility: 'auto',
              autoHide: 'scroll',
              autoHideDelay: 1300,
              theme: 'os-theme-dark',
            },
            overflow: {
              x: 'hidden',
            },
          }}
        >
          {props.children}
        </OverlayScrollbarsComponent>
      </Box>
    </Flex>
  );
};

export default memo(ScrollableContent);
