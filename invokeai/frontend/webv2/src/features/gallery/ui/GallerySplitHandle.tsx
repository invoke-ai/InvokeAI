import { Box, Flex } from '@chakra-ui/react';
import { useCallback, useMemo, type KeyboardEvent, type PointerEvent } from 'react';

const SPLIT_SIZE_STEP_PX = 16;
export const GALLERY_SPLIT_HANDLE_SIZE_PX = 8;

const HANDLE_HOVER_PROPS = { '& > div': { color: 'accent.solid' } } as const;
const HANDLE_FOCUS_PROPS = {
  '& > div': { color: 'accent.solid' },
  outline: '2px solid {colors.accent.solid}',
  outlineOffset: '-1px',
} as const;

// An 8px target painting a 1px line: a divider as thick as its grab area reads
// as a gap in the layout.
const HORIZONTAL_PROPS = {
  cursor: 'row-resize',
  h: `${GALLERY_SPLIT_HANDLE_SIZE_PX}px`,
  w: 'full',
} as const;
const VERTICAL_PROPS = {
  cursor: 'col-resize',
  h: 'full',
  w: `${GALLERY_SPLIT_HANDLE_SIZE_PX}px`,
} as const;

const HORIZONTAL_LINE_PROPS = { h: '1px', w: 'full' } as const;
const VERTICAL_LINE_PROPS = { h: 'full', w: '1px' } as const;

export const clampSplitSize = (sizePx: number, { max, min }: { max: number; min: number }): number =>
  Math.min(max, Math.max(min, Math.round(sizePx)));

/**
 * Resize handle for the board-panel split, serving both layouts. The parent
 * holds `onPreview`'s live value so a drag renders without writing settings on
 * every move; `onCommit` fires once, on release.
 */
export const GallerySplitHandle = ({
  label,
  max,
  min,
  orientation,
  sizePx,
  onCommit,
  onPreview,
}: {
  label: string;
  max: number;
  min: number;
  orientation: 'horizontal' | 'vertical';
  /** The size currently rendered, including any in-flight drag preview. */
  sizePx: number;
  onCommit: (sizePx: number) => void;
  onPreview: (sizePx: number | null) => void;
}) => {
  const isHorizontal = orientation === 'horizontal';

  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      event.preventDefault();

      const startX = event.clientX;
      const startY = event.clientY;
      const startSizePx = sizePx;
      let nextSizePx = startSizePx;
      const pointerSession = new AbortController();

      const handlePointerMove = (moveEvent: globalThis.PointerEvent) => {
        const deltaPx = isHorizontal ? moveEvent.clientY - startY : moveEvent.clientX - startX;

        nextSizePx = clampSplitSize(startSizePx + deltaPx, { max, min });
        onPreview(nextSizePx);
      };

      const handlePointerUp = () => {
        pointerSession.abort();
        onPreview(null);
        onCommit(nextSizePx);
      };

      window.addEventListener('pointermove', handlePointerMove, { signal: pointerSession.signal });
      window.addEventListener('pointerup', handlePointerUp, { signal: pointerSession.signal });
      window.addEventListener('pointercancel', handlePointerUp, { signal: pointerSession.signal });
    },
    [isHorizontal, max, min, sizePx, onCommit, onPreview]
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const step = event.shiftKey ? SPLIT_SIZE_STEP_PX * 2 : SPLIT_SIZE_STEP_PX;
      const sizeChanges: Partial<Record<string, number>> = isHorizontal
        ? { ArrowDown: step, ArrowUp: -step, End: max - sizePx, Home: min - sizePx }
        : { ArrowLeft: -step, ArrowRight: step, End: max - sizePx, Home: min - sizePx };
      const sizeChange = sizeChanges[event.key];

      if (sizeChange === undefined) {
        return;
      }

      event.preventDefault();
      onCommit(clampSplitSize(sizePx + sizeChange, { max, min }));
    },
    [isHorizontal, max, min, sizePx, onCommit]
  );

  const orientationProps = useMemo(() => (isHorizontal ? HORIZONTAL_PROPS : VERTICAL_PROPS), [isHorizontal]);

  return (
    <Flex
      align="center"
      aria-label={label}
      aria-orientation={isHorizontal ? 'horizontal' : 'vertical'}
      aria-valuemax={max}
      aria-valuemin={min}
      aria-valuenow={sizePx}
      flexShrink={0}
      justify="center"
      role="separator"
      tabIndex={0}
      _focusVisible={HANDLE_FOCUS_PROPS}
      _hover={HANDLE_HOVER_PROPS}
      {...orientationProps}
      onKeyDown={handleKeyDown}
      onPointerDown={handlePointerDown}
    >
      <Box
        bg="currentColor"
        color="border.subtle"
        transition="color var(--wb-motion-duration-fast) ease"
        {...(isHorizontal ? HORIZONTAL_LINE_PROPS : VERTICAL_LINE_PROPS)}
      />
    </Flex>
  );
};
