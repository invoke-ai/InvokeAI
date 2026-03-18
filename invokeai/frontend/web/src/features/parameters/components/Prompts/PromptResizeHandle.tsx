import { Box } from '@invoke-ai/ui-library';
import {
  memo,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

type PromptResizeHandleProps = {
  textareaRef: RefObject<HTMLTextAreaElement>;
  minHeight: number;
};

const PROMPT_RESIZE_HANDLE_HEIGHT_PX = 8;

export const PromptResizeHandle = memo(({ textareaRef, minHeight }: PromptResizeHandleProps) => {
  const activePointerIdRef = useRef<number | null>(null);
  const startHeightRef = useRef(0);
  const startYRef = useRef(0);
  const previousCursorRef = useRef('');
  const previousUserSelectRef = useRef('');
  const [isResizing, setIsResizing] = useState(false);

  const stopResize = useCallback(() => {
    if (activePointerIdRef.current === null) {
      return;
    }

    activePointerIdRef.current = null;
    setIsResizing(false);
    document.body.style.cursor = previousCursorRef.current;
    document.body.style.userSelect = previousUserSelectRef.current;
  }, []);

  useEffect(() => stopResize, [stopResize]);

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) {
        return;
      }

      const textarea = textareaRef.current;
      if (!textarea) {
        return;
      }

      activePointerIdRef.current = e.pointerId;
      startYRef.current = e.clientY;
      startHeightRef.current = textarea.offsetHeight;
      previousCursorRef.current = document.body.style.cursor;
      previousUserSelectRef.current = document.body.style.userSelect;

      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
      e.currentTarget.setPointerCapture(e.pointerId);
      setIsResizing(true);
      e.preventDefault();
    },
    [textareaRef]
  );

  const onPointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (activePointerIdRef.current !== e.pointerId) {
        return;
      }

      const textarea = textareaRef.current;
      if (!textarea) {
        return;
      }

      const nextHeight = Math.max(minHeight, startHeightRef.current + e.clientY - startYRef.current);
      textarea.style.height = `${nextHeight}px`;
      e.preventDefault();
    },
    [minHeight, textareaRef]
  );

  const onPointerUp = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (activePointerIdRef.current !== e.pointerId) {
        return;
      }

      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }

      stopResize();
    },
    [stopResize]
  );

  const onPointerCancel = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (activePointerIdRef.current !== e.pointerId) {
        return;
      }

      stopResize();
    },
    [stopResize]
  );

  return (
    <Box
      aria-hidden
      pos="absolute"
      insetInlineStart={0}
      insetInlineEnd={0}
      insetBlockEnd={0}
      h={`${PROMPT_RESIZE_HANDLE_HEIGHT_PX}px`}
      borderBottomRadius="base"
      bg={isResizing ? 'base.500' : 'base.700'}
      cursor="ns-resize"
      zIndex={1}
      style={{ touchAction: 'none' }}
      transitionProperty="background-color"
      transitionDuration="normal"
      _hover={{ bg: 'base.600' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onLostPointerCapture={stopResize}
    />
  );
});

PromptResizeHandle.displayName = 'PromptResizeHandle';
