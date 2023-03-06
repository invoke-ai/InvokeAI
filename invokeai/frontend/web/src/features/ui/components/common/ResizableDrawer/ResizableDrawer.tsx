import {
  Box,
  chakra,
  ChakraProps,
  Slide,
  useOutsideClick,
  useTheme,
  SlideDirection,
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
  getDefaultSize,
  getHandleEnables,
  getHandleStyles,
  getMinMaxDimensions,
  getResizableStyles,
} from './util';
import Scrollable from '../Scrollable';

type ResizableDrawerProps = ResizableProps & {
  children: ReactNode;
  isResizable: boolean;
  isPinned: boolean;
  isOpen: boolean;
  onClose: () => void;
  direction?: SlideDirection;
  initialWidth?: string | number;
  minWidth?: string | number;
  maxWidth?: string | number;
  initialHeight?: string | number;
  minHeight?: string | number;
  maxHeight?: string | number;
  shouldAllowResize?: boolean;
  onResizeStart?: ResizeStartCallback;
  onResizeStop?: ResizeCallback;
  onResize?: ResizeCallback;
  handleWidth?: number;
  handleInteractWidth?: string | number;
  sx?: ChakraProps['sx'];
  pinnedWidth: number;
  pinnedHeight: string | number;
};

const ChakraResizeable = chakra(Resizable, {
  shouldForwardProp: (prop) => !['sx'].includes(prop),
});

const ResizableDrawer = ({
  direction = 'left',
  isResizable,
  isPinned,
  isOpen,
  onClose,
  children,
  initialWidth = undefined,
  minWidth = undefined,
  maxWidth = undefined,
  initialHeight = undefined,
  minHeight = undefined,
  maxHeight = undefined,
  shouldAllowResize,
  onResizeStart,
  onResizeStop,
  onResize,
  handleWidth = 5,
  handleInteractWidth = '15px',
  pinnedWidth,
  pinnedHeight,
  sx = {},
}: ResizableDrawerProps) => {
  const langDirection = useTheme().direction as LangDirection;

  const outsideClickRef = useRef<HTMLDivElement>(null);

  useOutsideClick({
    ref: outsideClickRef,
    handler: () => {
      if (isPinned) {
        return;
      }

      onClose();
    },
  });

  const [width, setWidth] = useState<number | string>(0);
  const [height, setHeight] = useState<number | string>(0);

  const handleEnables = useMemo(
    () =>
      isResizable && shouldAllowResize
        ? getHandleEnables({ direction, langDirection })
        : {},
    [isResizable, shouldAllowResize, langDirection, direction]
  );

  const handleStyles = useMemo(
    () =>
      getHandleStyles({
        handleEnables,
        handleStyle: {
          width: handleInteractWidth,
        },
      }),
    [handleEnables, handleInteractWidth]
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

  const resizableStyles = useMemo(
    () => getResizableStyles({ isPinned, direction, sx, handleWidth }),
    [sx, handleWidth, direction, isPinned]
  );

  useEffect(() => {
    const { width, height } = getDefaultSize({
      initialWidth,
      initialHeight,
      direction,
    });

    setWidth(width);
    setHeight(height);
  }, [initialWidth, initialHeight, direction, langDirection]);

  useEffect(() => {
    if (['left', 'right'].includes(direction)) {
      setHeight(isPinned ? '100%' : '100vh');
    }
    if (['top', 'bottom'].includes(direction)) {
      setWidth(isPinned ? '100%' : '100vw');
    }
  }, [isPinned, direction]);

  return (
    <Slide
      direction={direction}
      in={isOpen}
      motionProps={{ initial: false }}
      {...(isPinned
        ? {
            style: {
              position: undefined,
              left: undefined,
              top: undefined,
              bottom: undefined,
              width: undefined,
            },
          }
        : {
            transition: { enter: { duration: 0.2 }, exit: { duration: 0.2 } },
            style: { zIndex: 98 },
          })}
    >
      <Box
        ref={outsideClickRef}
        sx={{
          width: ['left', 'right'].includes(direction) ? 'min-content' : 'full',
          height: ['left', 'right'].includes(direction)
            ? '100%'
            : 'min-content',
          position: 'relative',
        }}
      >
        <ChakraResizeable
          size={{
            width: isPinned ? '100%' : width,
            height: isPinned ? '100%' : height,
          }}
          enable={handleEnables}
          handleStyles={handleStyles}
          {...minMaxDimensions}
          sx={{ ...resizableStyles, height: 'full' }}
          onResizeStart={(event, direction, elementRef) => {
            onResizeStart && onResizeStart(event, direction, elementRef);
          }}
          onResize={(event, direction, elementRef, delta) => {
            onResize && onResize(event, direction, elementRef, delta);
          }}
          onResizeStop={(event, direction, elementRef, delta) => {
            event.stopPropagation();
            event.stopImmediatePropagation();
            event.preventDefault();
            if (direction === 'left' || direction === 'right') {
              setWidth(Number(width) + delta.width);
            }
            if (direction === 'top' || direction === 'bottom') {
              setHeight(Number(height) + delta.height);
            }
            onResizeStop && onResizeStop(event, direction, elementRef, delta);
          }}
        >
          <Scrollable>{children}</Scrollable>
        </ChakraResizeable>
      </Box>
    </Slide>
  );
};

export default ResizableDrawer;
