import { Box, defineStyleConfig, Flex, useStyleConfig } from '@chakra-ui/react';
import { memo } from 'react';
import type { PanelResizeHandleProps } from 'react-resizable-panels';
import { PanelResizeHandle } from 'react-resizable-panels';

type ResizeHandleProps = PanelResizeHandleProps & {
  orientation: 'horizontal' | 'vertical';
};

const ResizeHandle = (props: ResizeHandleProps) => {
  const { orientation, ...rest } = props;
  const styles = useStyleConfig('ResizeHandle', { orientation });

  return (
    <PanelResizeHandle {...rest}>
      <Flex __css={styles} data-orientation={orientation}>
        <Box className="resize-handle-inner" data-orientation={orientation} />
        <Box
          className="resize-handle-drag-handle"
          data-orientation={orientation}
        />
      </Flex>
    </PanelResizeHandle>
  );
};

export default memo(ResizeHandle);

export const resizeHandleTheme = defineStyleConfig({
  // The styles all Cards have in common
  baseStyle: () => ({
    display: 'flex',
    pos: 'relative',
    '&[data-orientation="horizontal"]': {
      w: 'full',
      h: 5,
    },
    '&[data-orientation="vertical"]': { w: 5, h: 'full' },
    alignItems: 'center',
    justifyContent: 'center',
    div: {
      bg: 'base.800',
    },
    _hover: {
      div: { bg: 'base.700' },
    },
    _active: {
      div: { bg: 'base.600' },
    },
    transitionProperty: 'common',
    transitionDuration: 'normal',
    '.resize-handle-inner': {
      '&[data-orientation="horizontal"]': {
        w: 'calc(100% - 1rem)',
        h: '2px',
      },
      '&[data-orientation="vertical"]': {
        w: '2px',
        h: 'calc(100% - 1rem)',
      },
      borderRadius: 'base',
      transitionProperty: 'inherit',
      transitionDuration: 'inherit',
    },
    '.resize-handle-drag-handle': {
      pos: 'absolute',
      borderRadius: '2px',
      transitionProperty: 'inherit',
      transitionDuration: 'inherit',
      '&[data-orientation="horizontal"]': {
        w: '20px',
        h: '6px',
        insetInlineStart: '50%',
        transform: 'translate(-50%, 0)',
      },
      '&[data-orientation="vertical"]': {
        w: '6px',
        h: '20px',
        insetBlockStart: '50%',
        transform: 'translate(0, -50%)',
      },
    },
  }),
});
