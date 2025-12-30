import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import type { CanvasTextSettingsState } from 'features/controlLayers/store/canvasTextSlice';
import { selectCanvasTextSlice } from 'features/controlLayers/store/canvasTextSlice';
import { getFontStackById, TEXT_RASTER_PADDING } from 'features/controlLayers/text/textConstants';
import { isAllowedTextShortcut } from 'features/controlLayers/text/textHotkeys';
import { hasVisibleGlyphs, measureTextContent, type TextMeasureConfig } from 'features/controlLayers/text/textRenderer';
import {
  type ClipboardEvent as ReactClipboardEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

export const CanvasTextOverlay = memo(() => {
  const canvasManager = useCanvasManager();
  const session = useStore(canvasManager.tool.tools.text.$session);
  const stageAttrs = useStore(canvasManager.stage.$stageAttrs);

  if (!session) {
    return null;
  }

  return (
    <Flex
      pointerEvents="none"
      position="absolute"
      inset={0}
      sx={{ isolation: 'isolate' }}
      data-testid="canvas-text-overlay"
    >
      <Box
        pointerEvents="none"
        position="absolute"
        inset={0}
        transform={`translate(${stageAttrs.x}px, ${stageAttrs.y}px) scale(${stageAttrs.scale})`}
        transformOrigin="top left"
      >
        <TextEditor sessionId={session.id} anchor={session.anchor} initialText={session.text} />
      </Box>
    </Flex>
  );
});

CanvasTextOverlay.displayName = 'CanvasTextOverlay';

const buildMeasureConfig = (text: string, settings: CanvasTextSettingsState): TextMeasureConfig => {
  const fontStyle: TextMeasureConfig['fontStyle'] = settings.italic ? 'italic' : 'normal';
  return {
    text,
    fontSize: settings.fontSize,
    fontFamily: getFontStackById(settings.fontId),
    fontWeight: settings.bold ? 700 : 400,
    fontStyle,
    lineHeight: settings.lineHeight,
  };
};

const TextEditor = ({
  sessionId,
  anchor,
  initialText,
}: {
  sessionId: string;
  anchor: { x: number; y: number };
  initialText: string;
}) => {
  const canvasManager = useCanvasManager();
  const textSettings = useAppSelector(selectCanvasTextSlice);
  const canvasSettings = useAppSelector(selectCanvasSettingsSlice);
  const editorRef = useRef<HTMLDivElement>(null);
  const lastSessionIdRef = useRef<string | null>(null);
  const hasFocusedRef = useRef(false);
  const focusAttemptIdRef = useRef<number | null>(null);
  const [isComposing, setIsComposing] = useState(false);
  const [textValue, setTextValue] = useState(initialText);
  const [contentMetrics, setContentMetrics] = useState(() =>
    measureTextContent(buildMeasureConfig(initialText, textSettings))
  );
  const [isEmpty, setIsEmpty] = useState(() => !hasVisibleGlyphs(initialText));

  const focusEditor = useCallback(() => {
    const node = editorRef.current;
    if (!node) {
      return;
    }
    node.focus({ preventScroll: true });
    const selection = window.getSelection();
    if (!selection) {
      return;
    }
    const range = document.createRange();
    range.selectNodeContents(node);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);
  }, []);

  useEffect(() => {
    const node = editorRef.current;
    if (!node) {
      return;
    }
    const isNewSession = lastSessionIdRef.current !== sessionId;
    if (isNewSession) {
      lastSessionIdRef.current = sessionId;
      hasFocusedRef.current = false;
      node.textContent = initialText;
      const syncedText = (node.innerText ?? '').replace(/\r/g, '');
      setIsEmpty(!hasVisibleGlyphs(syncedText));
      setTextValue(syncedText);
      setContentMetrics(measureTextContent(buildMeasureConfig(syncedText, textSettings)));
      canvasManager.tool.tools.text.updateSessionText(sessionId, syncedText);
    }

    if (hasFocusedRef.current) {
      return;
    }

    let attempts = 0;
    const MAX_ATTEMPTS = 5;
    const tryFocus = () => {
      const editorNode = editorRef.current;
      if (!editorNode) {
        return;
      }
      canvasManager.tool.tools.text.markSessionEditing(sessionId);
      focusEditor();
      const didFocus = document.activeElement === editorNode;
      attempts += 1;
      if (didFocus || attempts >= MAX_ATTEMPTS) {
        hasFocusedRef.current = true;
        focusAttemptIdRef.current = null;
        return;
      }
      focusAttemptIdRef.current = requestAnimationFrame(tryFocus);
    };

    focusAttemptIdRef.current = requestAnimationFrame(tryFocus);
    return () => {
      if (focusAttemptIdRef.current !== null) {
        cancelAnimationFrame(focusAttemptIdRef.current);
        focusAttemptIdRef.current = null;
      }
    };
  }, [canvasManager.tool.tools.text, focusEditor, initialText, sessionId, textSettings]);

  useEffect(() => {
    setContentMetrics(measureTextContent(buildMeasureConfig(textValue, textSettings)));
  }, [textSettings, textValue]);

  useEffect(() => {
    const shouldIgnorePointerDown = (event: PointerEvent) => {
      const target = event.target as HTMLElement | null;
      if (!target) {
        return false;
      }
      const path = event.composedPath?.() ?? [];
      for (const node of path) {
        if (!(node instanceof HTMLElement)) {
          continue;
        }
        const role = node.getAttribute('role');
        if (role === 'listbox' || role === 'option') {
          return true;
        }
        if (editorRef.current && editorRef.current.contains(node)) {
          return true;
        }
        if (node.dataset?.textToolSafezone === 'true') {
          return true;
        }
      }
      return editorRef.current?.contains(target) ?? false;
    };

    const handlePointerDown = (event: PointerEvent) => {
      if (shouldIgnorePointerDown(event)) {
        return;
      }
      canvasManager.tool.tools.text.requestCommit(sessionId);
    };
    window.addEventListener('pointerdown', handlePointerDown, true);
    return () => window.removeEventListener('pointerdown', handlePointerDown, true);
  }, [canvasManager.tool.tools.text, sessionId]);

  const handleInput = useCallback(() => {
    const value = (editorRef.current?.innerText ?? '').replace(/\r/g, '');
    setIsEmpty(!hasVisibleGlyphs(value));
    setTextValue(value);
    setContentMetrics(measureTextContent(buildMeasureConfig(value, textSettings)));
    canvasManager.tool.tools.text.updateSessionText(sessionId, value);
  }, [canvasManager.tool.tools.text, sessionId, textSettings]);

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const nativeEvent = event.nativeEvent;
      if (!isAllowedTextShortcut(nativeEvent)) {
        event.stopPropagation();
        nativeEvent.stopPropagation();
        nativeEvent.stopImmediatePropagation?.();
      }

      if (event.key === 'Enter' && !event.shiftKey && !isComposing) {
        event.preventDefault();
        canvasManager.tool.tools.text.requestCommit(sessionId);
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        canvasManager.tool.tools.text.clearSession();
      }
    },
    [canvasManager.tool.tools.text, isComposing, sessionId]
  );

  const handlePaste = useCallback((event: ReactClipboardEvent<HTMLDivElement>) => {
    event.preventDefault();
    const text = event.clipboardData.getData('text/plain');
    document.execCommand('insertText', false, text);
  }, []);

  const handleCompositionStart = useCallback(() => setIsComposing(true), []);
  const handleCompositionEnd = useCallback(() => setIsComposing(false), []);

  const containerMetrics = useMemo(() => {
    const padding = TEXT_RASTER_PADDING;
    const extraRightPadding = Math.ceil(textSettings.fontSize * 0.26);
    const extraLeftPadding = Math.ceil(textSettings.fontSize * 0.12);
    let offsetX = -padding - extraLeftPadding;
    if (textSettings.alignment === 'center') {
      offsetX = -(contentMetrics.contentWidth / 2) - padding - extraLeftPadding;
    } else if (textSettings.alignment === 'right') {
      offsetX = -contentMetrics.contentWidth - padding - extraLeftPadding;
    }
    return {
      x: anchor.x + offsetX,
      y: anchor.y - padding,
      padding,
      extraLeftPadding,
      extraRightPadding,
      width: contentMetrics.contentWidth + padding * 2 + extraLeftPadding + extraRightPadding,
      height: contentMetrics.contentHeight + padding * 2,
    };
  }, [
    anchor.x,
    anchor.y,
    contentMetrics.contentHeight,
    contentMetrics.contentWidth,
    textSettings.alignment,
    textSettings.fontSize,
  ]);

  useEffect(() => {
    canvasManager.tool.tools.text.updateSessionPosition(sessionId, {
      x: containerMetrics.x,
      y: containerMetrics.y,
    });
  }, [canvasManager.tool.tools.text, containerMetrics, sessionId]);

  const containerStyle = useMemo(() => {
    return {
      left: `${containerMetrics.x}px`,
      top: `${containerMetrics.y}px`,
      paddingTop: `${containerMetrics.padding}px`,
      paddingBottom: `${containerMetrics.padding}px`,
      paddingLeft: `${containerMetrics.padding + containerMetrics.extraLeftPadding}px`,
      paddingRight: `${containerMetrics.padding + containerMetrics.extraRightPadding}px`,
      width: `${Math.max(containerMetrics.width, textSettings.fontSize)}px`,
      height: `${Math.max(containerMetrics.height, textSettings.fontSize)}px`,
      textAlign: textSettings.alignment,
    };
  }, [containerMetrics, textSettings.alignment, textSettings.fontSize]);

  const textStyle = useMemo(() => {
    const color =
      canvasSettings.activeColor === 'fgColor'
        ? rgbaColorToString(canvasSettings.fgColor)
        : rgbaColorToString(canvasSettings.bgColor);
    const decorations: string[] = [];
    if (textSettings.underline) {
      decorations.push('underline');
    }
    if (textSettings.strikethrough) {
      decorations.push('line-through');
    }
    return {
      fontFamily: getFontStackById(textSettings.fontId),
      fontWeight: textSettings.bold ? 700 : 400,
      fontStyle: textSettings.italic ? 'italic' : 'normal',
      textDecorationLine: decorations.length ? decorations.join(' ') : 'none',
      fontSize: `${textSettings.fontSize}px`,
      lineHeight: textSettings.lineHeight,
      color,
      textAlign: textSettings.alignment,
    } as const;
  }, [canvasSettings, textSettings]);

  return (
    <Box position="absolute" pointerEvents="auto" {...containerStyle}>
      <Box
        ref={editorRef}
        contentEditable
        suppressContentEditableWarning
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        onCompositionStart={handleCompositionStart}
        onCompositionEnd={handleCompositionEnd}
        sx={{
          minWidth: '1ch',
          minHeight: `${textSettings.fontSize}px`,
          whiteSpace: 'pre',
          outline: 'none',
          cursor: 'text',
          display: 'inline-block',
          borderWidth: '1px',
          borderStyle: isEmpty ? 'dashed' : 'solid',
          borderColor: isEmpty ? 'baseAlpha.400' : 'transparent',
          borderRadius: 'sm',
          bg: isEmpty ? 'baseAlpha.200' : 'transparent',
          transitionProperty: 'border-color, background-color',
          transitionDuration: 'normal',
          ...textStyle,
        }}
      />
    </Box>
  );
};
