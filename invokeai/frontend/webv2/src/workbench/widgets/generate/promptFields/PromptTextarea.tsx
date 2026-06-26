import type { ResizableTextareaProps } from '@workbench/components/ui';
import type { PromptHighlightKind, PromptHighlightSegment } from '@workbench/generation/prompt/highlight';
import type { Ref, UIEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { ResizableTextarea } from '@workbench/components/ui';
import { buildPromptHighlightSegments } from '@workbench/generation/prompt/highlight';
import { useLayoutEffect, useMemo, useRef, useState } from 'react';

const MAX_HIGHLIGHTED_PROMPT_LENGTH = 20_000;
const PROMPT_TEXTAREA_LINE_HEIGHT = '1.6';
const PROMPT_TEXTAREA_PX = '2.5';
const PROMPT_TEXTAREA_PY = '2';

const HIGHLIGHT_STYLE_BY_KIND: Record<
  PromptHighlightKind,
  { bg?: string; color: string; textDecoration?: string; textDecorationColor?: string }
> = {
  attention: { color: 'accent.solid' },
  attentionNumeric: { color: 'fg.success' },
  embedding: { bg: 'bg.warning', color: 'fg.warning' },
  error: { bg: 'bg.error', color: 'fg.error', textDecoration: 'underline wavy', textDecorationColor: 'fg.error' },
  escapedParen: { color: 'fg.muted' },
  group: { color: 'fg.subtle' },
  promptFunctionArg: { bg: 'accent.subtle/20', color: 'fg' },
  promptFunctionMethod: { color: 'accent.fg' },
  punctuation: { color: 'fg.subtle' },
  text: { color: 'fg' },
};

interface PromptTextareaProps extends Omit<ResizableTextareaProps, 'underlay'> {
  showSyntaxHighlighting: boolean;
  value: string;
}

const setRef = (ref: Ref<HTMLTextAreaElement> | undefined, element: HTMLTextAreaElement | null): void => {
  if (!ref) {
    return;
  }

  if (typeof ref === 'function') {
    ref(element);
    return;
  }

  (ref as { current: HTMLTextAreaElement | null }).current = element;
};

const PromptHighlightSpan = ({ segment }: { segment: PromptHighlightSegment }) => {
  const style = HIGHLIGHT_STYLE_BY_KIND[segment.kind];

  return (
    <Box
      as="span"
      bg={style.bg}
      borderRadius={style.bg ? '2px' : undefined}
      color={style.color}
      textDecoration={style.textDecoration}
      textDecorationColor={style.textDecorationColor}
    >
      {segment.text}
    </Box>
  );
};

export const PromptTextarea = ({
  fontFamily = 'mono',
  fontSize,
  lineHeight,
  onScroll,
  showSyntaxHighlighting,
  textareaRef,
  value,
  ...props
}: PromptTextareaProps) => {
  const localTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [scroll, setScroll] = useState({ left: 0, top: 0 });
  const [textareaClientWidth, setTextareaClientWidth] = useState<number | null>(null);
  const shouldHighlight = showSyntaxHighlighting && value.length > 0 && value.length <= MAX_HIGHLIGHTED_PROMPT_LENGTH;
  const effectiveFontSize = fontSize ?? 'xs';
  const effectiveLineHeight = lineHeight ?? PROMPT_TEXTAREA_LINE_HEIGHT;
  const segments = useMemo(
    () => (shouldHighlight ? buildPromptHighlightSegments(value) : []),
    [shouldHighlight, value]
  );

  useLayoutEffect(() => {
    if (!shouldHighlight || !localTextareaRef.current) {
      return;
    }

    const textarea = localTextareaRef.current;
    const syncClientWidth = () => {
      const nextWidth = textarea.clientWidth;

      setTextareaClientWidth((currentWidth) => (currentWidth === nextWidth ? currentWidth : nextWidth));
    };
    const resizeObserver = new ResizeObserver(syncClientWidth);

    syncClientWidth();
    resizeObserver.observe(textarea);
    window.addEventListener('resize', syncClientWidth);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', syncClientWidth);
    };
  }, [shouldHighlight, value]);

  const handleTextareaRef = (element: HTMLTextAreaElement | null) => {
    localTextareaRef.current = element;
    setRef(textareaRef, element);
  };

  const handleScroll = (event: UIEvent<HTMLTextAreaElement>) => {
    setScroll({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop });
    onScroll?.(event);
  };

  const underlay = shouldHighlight ? (
    <Box
      aria-hidden="true"
      borderColor="transparent"
      borderRadius="md"
      borderWidth="1px"
      color="fg"
      inset="0"
      overflow="hidden"
      pointerEvents="none"
      position="absolute"
      zIndex={0}
      css={{ '@media (forced-colors: active)': { display: 'none' } }}
    >
      <Box
        as="pre"
        fontFamily={fontFamily}
        fontSize={effectiveFontSize}
        lineHeight={effectiveLineHeight}
        m="0"
        minH="100%"
        overflowWrap="break-word"
        px={PROMPT_TEXTAREA_PX}
        py={PROMPT_TEXTAREA_PY}
        transform={`translate(${-scroll.left}px, ${-scroll.top}px)`}
        whiteSpace="pre-wrap"
        w={textareaClientWidth ? `${textareaClientWidth}px` : '100%'}
      >
        {segments.map((segment) => (
          <PromptHighlightSpan key={`${segment.range.start}:${segment.range.end}:${segment.kind}`} segment={segment} />
        ))}
        {value.endsWith('\n') ? '\u200b' : null}
      </Box>
    </Box>
  ) : null;

  return (
    <ResizableTextarea
      {...props}
      bg={shouldHighlight ? 'transparent' : props.bg}
      caretColor={shouldHighlight ? 'fg' : props.caretColor}
      color={shouldHighlight ? 'transparent' : props.color}
      css={
        shouldHighlight
          ? {
              '&::selection': { color: 'transparent', WebkitTextFillColor: 'transparent' },
              '@media (forced-colors: active)': {
                WebkitTextFillColor: 'CanvasText',
                backgroundColor: 'Field',
                color: 'CanvasText',
              },
              WebkitTextFillColor: 'transparent',
            }
          : props.css
      }
      fontFamily={fontFamily}
      fontSize={effectiveFontSize}
      lineHeight={effectiveLineHeight}
      overflowWrap="break-word"
      px={PROMPT_TEXTAREA_PX}
      py={PROMPT_TEXTAREA_PY}
      textareaRef={handleTextareaRef}
      underlay={underlay}
      value={value}
      onScroll={handleScroll}
    />
  );
};
