/* eslint-disable react/react-compiler */
import type {
  ComponentProps,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  Ref,
  ReactNode,
} from 'react';

import { Box, Textarea } from '@chakra-ui/react';
import { useCallback, useState } from 'react';

type TextareaProps = ComponentProps<typeof Textarea>;

const DEFAULT_STEP_PX = 12;
const DEFAULT_LARGE_STEP_PX = 48;

const clamp = (value: number, min: number, max: number): number => Math.min(Math.max(value, min), max);

const resizeHandleAfter = {
  bg: 'border.emphasized',
  borderRadius: 'full',
  bottom: '1px',
  content: '""',
  h: '1px',
  left: '25%',
  opacity: 0.55,
  position: 'absolute',
  right: '25%',
} as const;

const resizeHandleFocusVisible = { bg: 'accent.solid/20', outline: '2px solid {colors.accent.solid}' } as const;
const resizeHandleHover = { bg: 'accent.solid/12' } as const;

export interface ResizableTextareaProps extends Omit<
  TextareaProps,
  'h' | 'height' | 'maxH' | 'maxHeight' | 'minH' | 'minHeight' | 'resize'
> {
  defaultHeightPx: number;
  maxHeightPx?: number;
  minHeightPx: number;
  resizeHandleAriaLabel: string;
  largeStepPx?: number;
  stepPx?: number;
  textareaRef?: Ref<HTMLTextAreaElement>;
  underlay?: ReactNode;
  onResizeEnd?: (heightPx: number) => void;
}

export const ResizableTextarea = ({
  defaultHeightPx,
  largeStepPx = DEFAULT_LARGE_STEP_PX,
  maxHeightPx = 420,
  minHeightPx,
  onResizeEnd,
  resizeHandleAriaLabel,
  stepPx = DEFAULT_STEP_PX,
  textareaRef,
  underlay,
  ...textareaProps
}: ResizableTextareaProps) => {
  const initialHeightPx = clamp(defaultHeightPx, minHeightPx, maxHeightPx);
  const [heightPx, setHeightPx] = useState(initialHeightPx);
  const [dragHeightPx, setDragHeightPx] = useState<number | null>(null);
  const displayHeightPx = dragHeightPx ?? heightPx;

  const commitHeight = useCallback(
    (nextHeightPx: number) => {
      const clampedHeightPx = clamp(nextHeightPx, minHeightPx, maxHeightPx);

      setHeightPx(clampedHeightPx);
      onResizeEnd?.(clampedHeightPx);
    },
    [maxHeightPx, minHeightPx, onResizeEnd]
  );

  const handlePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.preventDefault();

      const startY = event.clientY;
      const startHeightPx = displayHeightPx;
      let nextHeightPx = startHeightPx;

      const handlePointerMove = (moveEvent: PointerEvent) => {
        nextHeightPx = clamp(startHeightPx + moveEvent.clientY - startY, minHeightPx, maxHeightPx);
        setDragHeightPx(nextHeightPx);
      };

      const handlePointerUp = () => {
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
        window.removeEventListener('pointercancel', handlePointerUp);
        setDragHeightPx(null);
        commitHeight(nextHeightPx);
      };

      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
      window.addEventListener('pointercancel', handlePointerUp);
    },
    [commitHeight, displayHeightPx, maxHeightPx, minHeightPx]
  );

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? largeStepPx : stepPx;
      const heightChange =
        event.key === 'ArrowDown'
          ? step
          : event.key === 'ArrowUp'
            ? -step
            : event.key === 'End'
              ? maxHeightPx - displayHeightPx
              : event.key === 'Home'
                ? minHeightPx - displayHeightPx
                : undefined;

      if (heightChange === undefined) {
        return;
      }

      event.preventDefault();
      commitHeight(displayHeightPx + heightChange);
    },
    [commitHeight, displayHeightPx, largeStepPx, maxHeightPx, minHeightPx, stepPx]
  );

  return (
    <Box position="relative">
      {underlay}
      <Textarea
        ref={textareaRef}
        h={`${displayHeightPx}px`}
        position={underlay ? 'relative' : textareaProps.position}
        resize="none"
        zIndex={underlay ? 1 : textareaProps.zIndex}
        {...textareaProps}
      />
      <Box
        aria-label={resizeHandleAriaLabel}
        aria-orientation="horizontal"
        aria-valuemax={maxHeightPx}
        aria-valuemin={minHeightPx}
        aria-valuenow={displayHeightPx}
        bottom="0"
        cursor="ns-resize"
        h="2"
        left="0"
        position="absolute"
        right="0"
        role="separator"
        tabIndex={0}
        transition="background var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease"
        zIndex={2}
        _after={resizeHandleAfter}
        _focusVisible={resizeHandleFocusVisible}
        _hover={resizeHandleHover}
        onKeyDown={handleKeyDown}
        onPointerDown={handlePointerDown}
      />
    </Box>
  );
};
