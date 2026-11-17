import type {
  ComponentProps,
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
  Ref,
  ReactNode,
} from 'react';

import { Box, ScrollArea, Textarea } from '@chakra-ui/react';
import { useCallback, useState } from 'react';

type TextareaProps = ComponentProps<typeof Textarea>;

const DEFAULT_STEP_PX = 12;
const DEFAULT_LARGE_STEP_PX = 48;

const clamp = (value: number, min: number, max?: number): number => {
  const minClamped = Math.max(value, min);

  return max === undefined ? minClamped : Math.min(minClamped, max);
};

const resizeHandleAfter = {
  bg: 'border.emphasized',
  borderRadius: 'full',
  bottom: '3px',
  content: '""',
  h: '3px',
  left: '50%',
  position: 'absolute',
  transform: 'translateX(-50%)',
  transition: 'background var(--wb-motion-duration-fast) ease',
  w: '10',
} as const;

/*
 * The content under the handle fades out. A host that paints its controls on an opaque well marks itself with
 * `data-control-well` and sets the colour as `--wb-control-well`; the fade then ends in that colour. Any other host
 * gets a mask that fades the control to whatever lies behind it, and its edge strips keep the border and the rounded
 * corners opaque.
 *
 * The well's fade is an overlay rather than a mask because of scaled hosts such as the node editor. There the text
 * area's inner bottom edge falls inside a device pixel, and that pixel row holds both the bottom border and text the
 * clip lets through. A mask can only show or hide the whole row, and a fill that stops at the inner edge leaves a
 * share of the row's text uncovered, so either way a slice of text shows below the fade. The overlay instead covers
 * the bottom 1rem, border included, and repaints the border on top. Its fill is solid for the last 0.25rem.
 */
const resizeHandleFade = {
  maskImage: [
    'linear-gradient(to bottom, black calc(100% - 1rem), transparent calc(100% - 0.25rem), transparent calc(100% - 1px), black calc(100% - 1px))',
    'linear-gradient(to right, black {radii.control}, transparent {radii.control}, transparent calc(100% - {radii.control}), black calc(100% - {radii.control}))',
  ].join(', '),
  '[data-control-well] &': { maskImage: 'none' },
} as const;

// The fill covers the border's rows too. The border goes back on top as a separate box: Chromium draws a gradient fill
// short of a box's own opaque border, which would uncover that row again. The border follows the text area's own
// border states (`formControlInteraction` in the theme recipes), later entries winning as they do there.
const wellFade = {
  backgroundImage: 'linear-gradient(to bottom, transparent, var(--wb-control-well) calc(100% - 0.25rem))',
  borderBottomRadius: 'control',
  bottom: '0',
  display: 'none',
  h: '1rem',
  insetX: '0',
  pointerEvents: 'none',
  position: 'absolute',
  _after: {
    borderBottomRadius: 'inherit',
    borderColor: 'border',
    borderStyle: 'solid',
    borderWidth: '0 1px 1px',
    content: '""',
    inset: '0',
    position: 'absolute',
    transitionDuration: 'fast',
    transitionProperty: 'border-color',
  },
  '[data-control-well] &': { display: 'block' },
  '[data-part="viewport"]:hover ~ &::after': { borderColor: 'border.emphasized' },
  '[data-part="viewport"]:focus-visible ~ &::after': { borderColor: 'accent.solid' },
  '[data-part="viewport"]:is([aria-invalid="true"], [data-invalid]) ~ &::after': { borderColor: 'border.error' },
} as const;

const resizeHandleFocusVisible = { bg: 'accent.solid/20', outline: '2px solid {colors.accent.solid}' } as const;
const resizeHandleHover = { _after: { bg: 'fg.subtle' } } as const;
const resizeHandleDragging = { '&[data-dragging]::after': { bg: 'fg.subtle' } } as const;

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
  maxHeightPx,
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
              ? maxHeightPx === undefined
                ? undefined
                : maxHeightPx - displayHeightPx
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
      <Box position="relative" css={resizeHandleFade}>
        {underlay}
        <ScrollArea.Root
          borderRadius="control"
          h={`${displayHeightPx}px`}
          variant="hover"
          zIndex={underlay ? 1 : undefined}
        >
          <ScrollArea.Viewport asChild role="textbox">
            <Textarea ref={textareaRef} resize="none" {...textareaProps} />
          </ScrollArea.Viewport>
          {/* Zag needs a content element to remeasure; the textarea itself is the viewport. */}
          <ScrollArea.Content h="0" minW="0" overflow="hidden" position="absolute" />
          <Box aria-hidden data-resizable-textarea-fade="" css={wellFade} />
          <ScrollArea.Scrollbar>
            <ScrollArea.Thumb />
          </ScrollArea.Scrollbar>
        </ScrollArea.Root>
      </Box>
      <Box
        aria-label={resizeHandleAriaLabel}
        aria-orientation="horizontal"
        aria-valuemax={maxHeightPx}
        aria-valuemin={minHeightPx}
        aria-valuenow={displayHeightPx}
        bottom="0"
        cursor="ns-resize"
        data-dragging={dragHeightPx === null ? undefined : ''}
        h="2.5"
        left="0"
        position="absolute"
        right="0"
        role="separator"
        tabIndex={0}
        transition="background var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease"
        zIndex={2}
        css={resizeHandleDragging}
        _after={resizeHandleAfter}
        _focusVisible={resizeHandleFocusVisible}
        _hover={resizeHandleHover}
        onKeyDown={handleKeyDown}
        onPointerDown={handlePointerDown}
      />
    </Box>
  );
};
