import type { ResizableTextareaProps } from '@platform/ui';
import type { Ref, UIEvent } from 'react';

import { Box } from '@chakra-ui/react';
import { HighlightedPrompt, MAX_HIGHLIGHTED_PROMPT_LENGTH } from '@features/generation/ui/promptFields/PromptHighlight';
import { getLineNumberGutterCh, PromptLineNumbers } from '@features/generation/ui/promptFields/PromptLineNumbers';
import { ResizableTextarea } from '@platform/ui';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
const PROMPT_TEXTAREA_LINE_HEIGHT = '1.6';
// Literal lengths rather than spacing tokens: the gutter offset is a `calc()`,
// and Chakra's generated var name for a fractional token does not resolve inside
// one, which silently invalidates the whole declaration. These are the `2.5` and
// `2` spacing tokens.
const PROMPT_TEXTAREA_PX = '0.625rem';
const PROMPT_TEXTAREA_PY = '0.5rem';

const PROMPT_TEXTAREA_FORCED_COLORS_CSS = { '@media (forced-colors: active)': { display: 'none' } };
const PROMPT_TEXTAREA_HIGHLIGHTED_CSS = {
  '&::selection': { color: 'transparent', WebkitTextFillColor: 'transparent' },
  '@media (forced-colors: active)': {
    WebkitTextFillColor: 'CanvasText',
    backgroundColor: 'Field',
    color: 'CanvasText',
  },
  WebkitTextFillColor: 'transparent',
};

interface PromptTextareaProps extends Omit<ResizableTextareaProps, 'underlay'> {
  /**
   * Annotate `{a|b}` syntax. Only surfaces whose prompt is batch-expanded set
   * this — elsewhere the braces are literal text and colouring them would
   * promise an expansion that never happens.
   */
  highlightDynamicPrompts?: boolean;
  /** Resolvable wildcard names; an unknown `__name__` is underlined as an error. */
  knownWildcards?: ReadonlySet<string>;
  showSyntaxHighlighting: boolean;
  /** Number logical lines in a leading gutter, the way an editor does. */
  showLineNumbers?: boolean;
  /**
   * `[before, authored, after]` from `getPromptTemplateChunks`. When set, the
   * mirror dims the outer chunks so the template's own words read as context
   * around the user's text. `value` must be the concatenation of the three, and
   * the textarea is expected to be `readOnly` — this is a view, not an editor.
   */
  templateChunks?: [string, string, string] | null;
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

export const PromptTextarea = ({
  fontFamily = 'mono',
  fontSize,
  highlightDynamicPrompts = false,
  knownWildcards,
  lineHeight,
  onScroll,
  showLineNumbers = false,
  showSyntaxHighlighting,
  templateChunks = null,
  textareaRef,
  value,
  ...props
}: PromptTextareaProps) => {
  const localTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [scroll, setScroll] = useState({ left: 0, top: 0 });
  const [textareaClientWidth, setTextareaClientWidth] = useState<number | null>(null);
  const isWithinHighlightBudget = value.length > 0 && value.length <= MAX_HIGHLIGHTED_PROMPT_LENGTH;
  // Template chunks are a legibility device, not a syntax preference, so they
  // render whether or not highlighting is switched on.
  const shouldHighlight = (showSyntaxHighlighting || templateChunks !== null) && isWithinHighlightBudget;
  const effectiveFontSize = fontSize ?? '0.82rem';
  const effectiveLineHeight = lineHeight ?? PROMPT_TEXTAREA_LINE_HEIGHT;
  // The gutter widens with the line count, and the text has to start clear of it.
  const gutterCh = showLineNumbers ? getLineNumberGutterCh(value.split('\n').length) : 0;
  const paddingInlineStart = showLineNumbers ? `calc(${PROMPT_TEXTAREA_PX} + ${gutterCh}ch)` : undefined;

  const highlightOptions = useMemo(
    () => ({ dynamicPrompts: highlightDynamicPrompts, knownWildcards }),
    [highlightDynamicPrompts, knownWildcards]
  );

  const needsMirror = shouldHighlight || showLineNumbers;

  useLayoutEffect(() => {
    if (!needsMirror || !localTextareaRef.current) {
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
  }, [needsMirror, value]);

  const handleTextareaRef = useCallback(
    (element: HTMLTextAreaElement | null) => {
      localTextareaRef.current = element;
      setRef(textareaRef, element);
    },
    [textareaRef]
  );

  const handleScroll = useCallback(
    (event: UIEvent<HTMLTextAreaElement>) => {
      setScroll({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop });
      onScroll?.(event);
    },
    [onScroll]
  );

  const lineNumberMetrics = useMemo(
    () => ({
      clientWidth: textareaClientWidth,
      fontFamily,
      fontSize: effectiveFontSize,
      lineHeight: effectiveLineHeight,
      paddingBlock: PROMPT_TEXTAREA_PY,
      paddingInline: PROMPT_TEXTAREA_PX,
      scrollTop: scroll.top,
    }),
    [effectiveFontSize, effectiveLineHeight, fontFamily, scroll.top, textareaClientWidth]
  );

  const highlightUnderlay = useMemo(
    () =>
      shouldHighlight ? (
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
          css={PROMPT_TEXTAREA_FORCED_COLORS_CSS}
        >
          <Box
            as="pre"
            fontFamily={fontFamily}
            fontSize={effectiveFontSize}
            lineHeight={effectiveLineHeight}
            m="0"
            minH="100%"
            overflowWrap="break-word"
            paddingBlock={PROMPT_TEXTAREA_PY}
            paddingInline={PROMPT_TEXTAREA_PX}
            paddingInlineStart={paddingInlineStart}
            transform={`translate(${-scroll.left}px, ${-scroll.top}px)`}
            whiteSpace="pre-wrap"
            w={textareaClientWidth ? `${textareaClientWidth}px` : '100%'}
          >
            {templateChunks ? (
              <>
                <Box as="span" color="fg.subtle">
                  {templateChunks[0]}
                </Box>
                {showSyntaxHighlighting ? (
                  <HighlightedPrompt options={highlightOptions} prompt={templateChunks[1]} />
                ) : (
                  templateChunks[1]
                )}
                <Box as="span" color="fg.subtle">
                  {templateChunks[2]}
                </Box>
              </>
            ) : (
              <HighlightedPrompt options={highlightOptions} prompt={value} />
            )}
            {value.endsWith('\n') ? '\u200b' : null}
          </Box>
        </Box>
      ) : null,
    [
      effectiveFontSize,
      effectiveLineHeight,
      fontFamily,
      highlightOptions,
      paddingInlineStart,
      scroll.left,
      scroll.top,
      shouldHighlight,
      showSyntaxHighlighting,
      templateChunks,
      textareaClientWidth,
      value,
    ]
  );

  const underlay = useMemo(
    () =>
      needsMirror ? (
        <>
          {highlightUnderlay}
          {showLineNumbers ? <PromptLineNumbers metrics={lineNumberMetrics} value={value} /> : null}
        </>
      ) : null,
    [highlightUnderlay, lineNumberMetrics, needsMirror, showLineNumbers, value]
  );

  return (
    <ResizableTextarea
      {...props}
      bg={shouldHighlight ? 'transparent' : props.bg}
      caretColor={shouldHighlight ? 'fg' : props.caretColor}
      color={shouldHighlight ? 'transparent' : props.color}
      css={shouldHighlight ? PROMPT_TEXTAREA_HIGHLIGHTED_CSS : props.css}
      fontFamily={fontFamily}
      fontSize={effectiveFontSize}
      lineHeight={effectiveLineHeight}
      overflowWrap="break-word"
      paddingBlock={PROMPT_TEXTAREA_PY}
      paddingInline={PROMPT_TEXTAREA_PX}
      paddingInlineStart={paddingInlineStart}
      textareaRef={handleTextareaRef}
      underlay={underlay}
      value={value}
      onScroll={handleScroll}
    />
  );
};
