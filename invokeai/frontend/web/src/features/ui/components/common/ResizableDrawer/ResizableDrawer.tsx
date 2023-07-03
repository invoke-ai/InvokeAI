import {
  Box,
  chakra,
  ChakraProps,
  Slide,
  useOutsideClick,
  useTheme,
  SlideDirection,
  useColorMode,
} from '@chakra-ui/react';
import {
  Resizable,
  ResizableProps,
  ResizeCallback,
  ResizeStartCallback,
} from 're-resizable';
import { ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { LangDirection } from './types';
import {
  getHandleEnables,
  getMinMaxDimensions,
  getSlideDirection,
  getStyles,
} from './util';
import { mode } from 'theme/util/mode';

type ResizableDrawerProps = ResizableProps & {
  children: ReactNode;
  isResizable: boolean;
  isOpen: boolean;
  onClose: () => void;
  direction?: SlideDirection;
  initialWidth?: number;
  minWidth?: number;
  maxWidth?: number;
  initialHeight?: number;
  minHeight?: number;
  maxHeight?: number;
  onResizeStart?: ResizeStartCallback;
  onResizeStop?: ResizeCallback;
  onResize?: ResizeCallback;
  handleWidth?: string | number;
  handleInteractWidth?: string | number;
  sx?: ChakraProps['sx'];
};

const ChakraResizeable = chakra(Resizable, {
  shouldForwardProp: (prop) => !['sx'].includes(prop),
});

const ResizableDrawer = ({
  direction = 'left',
  isResizable,
  isOpen,
  onClose,
  children,
  initialWidth,
  minWidth,
  maxWidth,
  initialHeight,
  minHeight,
  maxHeight,
  onResizeStart,
  onResizeStop,
  onResize,
  sx = {},
}: ResizableDrawerProps) => {
  const langDirection = useTheme().direction as LangDirection;
  const { colorMode } = useColorMode();
  const outsideClickRef = useRef<HTMLDivElement>(null);

  const defaultWidth = useMemo(
    () =>
      initialWidth ??
      minWidth ??
      (['left', 'right'].includes(direction) ? 'auto' : '100%'),
    [initialWidth, minWidth, direction]
  );

  const defaultHeight = useMemo(
    () =>
      initialHeight ??
      minHeight ??
      (['top', 'bottom'].includes(direction) ? 'auto' : '100%'),
    [initialHeight, minHeight, direction]
  );

  const [width, setWidth] = useState<number | string>(defaultWidth);

  const [height, setHeight] = useState<number | string>(defaultHeight);

  useOutsideClick({
    ref: outsideClickRef,
    handler: () => {
      onClose();
    },
    enabled: isOpen,
  });

  const handleEnables = useMemo(
    () => (isResizable ? getHandleEnables({ direction, langDirection }) : {}),
    [isResizable, langDirection, direction]
  );

  const minMaxDimensions = useMemo(
    () =>
      getMinMaxDimensions({
        direction,
        minWidth,
        maxWidth,
        minHeight,
        maxHeight,
      }),
    [minWidth, maxWidth, minHeight, maxHeight, direction]
  );

  const { containerStyles, handleStyles } = useMemo(
    () =>
      getStyles({
        isResizable,
        direction,
      }),
    [isResizable, direction]
  );

  const slideDirection = useMemo(
    () => getSlideDirection(direction, langDirection),
    [direction, langDirection]
  );

  useEffect(() => {
    if (['left', 'right'].includes(direction)) {
      setHeight('100vh');
      // setHeight(isPinned ? '100%' : '100vh');
    }
    if (['top', 'bottom'].includes(direction)) {
      setWidth('100vw');
      // setWidth(isPinned ? '100%' : '100vw');
    }
  }, [direction]);

  return (
    <Slide
      direction={slideDirection}
      in={isOpen}
      motionProps={{ initial: false }}
      style={{ width: 'full' }}
    >
      <Box
        ref={outsideClickRef}
        sx={{
          width: 'full',
          height: 'full',
        }}
      >
        <ChakraResizeable
          size={{
            width: isResizable ? width : defaultWidth,
            height: isResizable ? height : defaultHeight,
          }}
          enable={handleEnables}
          handleStyles={handleStyles}
          {...minMaxDimensions}
          sx={{
            borderColor: mode('base.200', 'base.800')(colorMode),
            p: 4,
            bg: mode('base.100', 'base.900')(colorMode),
            height: 'full',
            shadow: isOpen ? 'dark-lg' : undefined,
            ...containerStyles,
            ...sx,
          }}
          onResizeStart={(event, direction, elementRef) => {
            onResizeStart && onResizeStart(event, direction, elementRef);
          }}
          onResize={(event, direction, elementRef, delta) => {
            onResize && onResize(event, direction, elementRef, delta);
          }}
          onResizeStop={(event, direction, elementRef, delta) => {
            if (['left', 'right'].includes(direction)) {
              setWidth(Number(width) + delta.width);
            }
            if (['top', 'bottom'].includes(direction)) {
              setHeight(Number(height) + delta.height);
            }
            onResizeStop && onResizeStop(event, direction, elementRef, delta);
          }}
        >
          {children}
        </ChakraResizeable>
      </Box>
    </Slide>
  );
};

export default ResizableDrawer;
