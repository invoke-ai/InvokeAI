import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import type { CanvasTextSettingsState } from 'features/controlLayers/store/canvasTextSlice';
import { selectCanvasTextSlice } from 'features/controlLayers/store/canvasTextSlice';
import type { Coordinate } from 'features/controlLayers/store/types';
import { getFontStackById, TEXT_RASTER_PADDING } from 'features/controlLayers/text/textConstants';
import { isAllowedTextShortcut } from 'features/controlLayers/text/textHotkeys';
import { measureTextContent, type TextMeasureConfig } from 'features/controlLayers/text/textRenderer';
import {
  type ClipboardEvent as ReactClipboardEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
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
        <TextEditor sessionId={session.id} anchor={session.anchor} initialText={session.text} stageAttrs={stageAttrs} />
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
  stageAttrs,
}: {
  sessionId: string;
  anchor: { x: number; y: number };
  initialText: string;
  stageAttrs: { x: number; y: number; scale: number };
}) => {
  const canvasManager = useCanvasManager();
  const textSettings = useAppSelector(selectCanvasTextSlice);
  const canvasSettings = useAppSelector(selectCanvasSettingsSlice);
  const editorRef = useRef<HTMLDivElement | null>(null);
  const lastSessionIdRef = useRef<string | null>(null);
  const lastFocusedSessionIdRef = useRef<string | null>(null);
  const focusRafIdRef = useRef<number | null>(null);
  const [isComposing, setIsComposing] = useState(false);
  const [textValue, setTextValue] = useState(initialText);
  const [contentMetrics, setContentMetrics] = useState(() =>
    measureTextContent(buildMeasureConfig(initialText, textSettings))
  );
  const dragStateRef = useRef<{
    pointerId: number;
    startPointer: Coordinate;
    startAnchor: Coordinate;
  } | null>(null);
  const [isCtrlPressed, setIsCtrlPressed] = useState(false);

  const isMoveModifierPressed = useCallback((event: { ctrlKey: boolean; metaKey: boolean }) => {
    return event.ctrlKey || event.metaKey;
  }, []);

  const getStagePoint = useCallback(
    (event: ReactPointerEvent<HTMLElement>) => {
      const rect = canvasManager.stage.container.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left - stageAttrs.x) / stageAttrs.scale,
        y: (event.clientY - rect.top - stageAttrs.y) / stageAttrs.scale,
      };
    },
    [canvasManager.stage.container, stageAttrs.x, stageAttrs.y, stageAttrs.scale]
  );

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

  const setEditorRef = useCallback((node: HTMLDivElement | null) => {
    editorRef.current = node;
  }, []);

  useEffect(() => {
    const node = editorRef.current;
    if (!node) {
      return;
    }
    const isNewSession = lastSessionIdRef.current !== sessionId;
    if (isNewSession) {
      lastSessionIdRef.current = sessionId;
      lastFocusedSessionIdRef.current = null;
      node.textContent = initialText;
      const syncedText = (node.innerText ?? '').replace(/\r/g, '');
      setTextValue(syncedText);
      setContentMetrics(measureTextContent(buildMeasureConfig(syncedText, textSettings)));
      canvasManager.tool.tools.text.updateSessionText(sessionId, syncedText);
    }
    if (lastFocusedSessionIdRef.current !== sessionId) {
      if (focusRafIdRef.current !== null) {
        cancelAnimationFrame(focusRafIdRef.current);
      }
      focusRafIdRef.current = requestAnimationFrame(() => {
        canvasManager.tool.tools.text.markSessionEditing(sessionId);
        focusEditor();
        lastFocusedSessionIdRef.current = sessionId;
        focusRafIdRef.current = null;
      });
    }
    return () => {
      if (focusRafIdRef.current !== null) {
        cancelAnimationFrame(focusRafIdRef.current);
        focusRafIdRef.current = null;
      }
    };
  }, [canvasManager.tool.tools.text, focusEditor, initialText, sessionId, textSettings]);

  useEffect(() => {
    setContentMetrics(measureTextContent(buildMeasureConfig(textValue, textSettings)));
  }, [textSettings, textValue]);

  const handleInput = useCallback(() => {
    const value = (editorRef.current?.innerText ?? '').replace(/\r/g, '');
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

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Control' || event.key === 'Meta') {
        setIsCtrlPressed(true);
      }
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === 'Control' || event.key === 'Meta') {
        setIsCtrlPressed(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  const handleContainerPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const modifierPressed = isMoveModifierPressed(event);
      if (modifierPressed !== isCtrlPressed) {
        setIsCtrlPressed(modifierPressed);
      }
      if (!modifierPressed) {
        if (canvasManager.tool.$tool.get() !== 'text') {
          canvasManager.tool.$tool.set('text');
        }
        return;
      }
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const startPointer = getStagePoint(event);
      dragStateRef.current = {
        pointerId: event.pointerId,
        startPointer,
        startAnchor: { ...anchor },
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [anchor, getStagePoint, isCtrlPressed, isMoveModifierPressed]
  );

  const handleContainerPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const modifierPressed = isMoveModifierPressed(event);
      if (modifierPressed !== isCtrlPressed) {
        setIsCtrlPressed(modifierPressed);
      }
      const dragState = dragStateRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) {
        return;
      }
      event.preventDefault();
      const currentPointer = getStagePoint(event);
      const deltaX = currentPointer.x - dragState.startPointer.x;
      const deltaY = currentPointer.y - dragState.startPointer.y;
      canvasManager.tool.tools.text.updateSessionAnchor(sessionId, {
        x: dragState.startAnchor.x + deltaX,
        y: dragState.startAnchor.y + deltaY,
      });
    },
    [canvasManager.tool.tools.text, getStagePoint, isCtrlPressed, isMoveModifierPressed, sessionId]
  );

  const handleContainerPointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) {
      return;
    }
    event.currentTarget.releasePointerCapture(event.pointerId);
    dragStateRef.current = null;
  }, []);

  const textContainerData = useMemo(() => {
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
      x: textContainerData.x,
      y: textContainerData.y,
    });
  }, [canvasManager.tool.tools.text, sessionId, textContainerData]);

  const containerStyle = useMemo(() => {
    return {
      left: `${textContainerData.x}px`,
      top: `${textContainerData.y}px`,
      paddingTop: `${textContainerData.padding}px`,
      paddingBottom: `${textContainerData.padding}px`,
      paddingLeft: `${textContainerData.padding + textContainerData.extraLeftPadding}px`,
      paddingRight: `${textContainerData.padding + textContainerData.extraRightPadding}px`,
      width: `${Math.max(textContainerData.width, textSettings.fontSize)}px`,
      height: `${Math.max(textContainerData.height, textSettings.fontSize)}px`,
      textAlign: textSettings.alignment,
    };
  }, [textContainerData, textSettings.alignment, textSettings.fontSize]);

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
      lineHeight: `${contentMetrics.lineHeightPx}px`,
      color,
      textAlign: textSettings.alignment,
    } as const;
  }, [canvasSettings, contentMetrics.lineHeightPx, textSettings]);

  return (
    <Box
      position="absolute"
      pointerEvents="auto"
      borderWidth="1px"
      borderStyle="dotted"
      borderColor="invokeBlue.300"
      borderRadius="md"
      boxSizing="border-box"
      sx={{ cursor: isCtrlPressed ? 'move' : 'text' }}
      onPointerDown={handleContainerPointerDown}
      onPointerMove={handleContainerPointerMove}
      onPointerUp={handleContainerPointerUp}
      onPointerCancel={handleContainerPointerUp}
      {...containerStyle}
    >
      <Box
        ref={setEditorRef}
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
          cursor: isCtrlPressed ? 'move' : 'text',
          display: 'inline-block',
          pointerEvents: isCtrlPressed ? 'none' : 'auto',
          ...textStyle,
        }}
      />
    </Box>
  );
};
