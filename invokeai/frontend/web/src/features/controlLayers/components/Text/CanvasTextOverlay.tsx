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
  memo,
  type PointerEvent as ReactPointerEvent,
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
        <TextEditor
          sessionId={session.id}
          anchor={session.anchor}
          initialText={session.text}
          rotation={session.rotation}
          stageAttrs={stageAttrs}
        />
      </Box>
    </Flex>
  );
});

CanvasTextOverlay.displayName = 'CanvasTextOverlay';

const ROTATE_ANCHOR_SIZE = 14;
const ROTATE_ANCHOR_BASE_SIZE = 12;
// Match Konva.Transformer default rotate anchor offset (50px) for consistent spacing.
const ROTATE_ANCHOR_GAP = 50;
const ROTATE_ANCHOR_LINE_LENGTH = ROTATE_ANCHOR_GAP;
const ROTATE_ANCHOR_FILL = 'invokeBlue.50';
const ROTATE_ANCHOR_STROKE = 'invokeBlue.500';

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
  rotation,
  stageAttrs,
}: {
  sessionId: string;
  anchor: { x: number; y: number };
  initialText: string;
  rotation: number;
  stageAttrs: { x: number; y: number; scale: number };
}) => {
  const canvasManager = useCanvasManager();
  const textSettings = useAppSelector(selectCanvasTextSlice);
  const canvasSettings = useAppSelector(selectCanvasSettingsSlice);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const editorRef = useRef<HTMLDivElement | null>(null);
  const lastSessionIdRef = useRef<string | null>(null);
  const lastFocusedSessionIdRef = useRef<string | null>(null);
  const focusRafIdRef = useRef<number | null>(null);
  const measureRafIdRef = useRef<number | null>(null);
  const lastObservedSizeRef = useRef<{ width: number; height: number } | null>(null);
  const lastMeasuredSizeRef = useRef<{ width: number; height: number } | null>(null);
  const [isComposing, setIsComposing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [textValue, setTextValue] = useState(initialText);
  const [contentMetrics, setContentMetrics] = useState(() =>
    measureTextContent(buildMeasureConfig(initialText, textSettings))
  );
  const [measuredSize, setMeasuredSize] = useState(() => ({
    width: Math.max(contentMetrics.contentWidth, textSettings.fontSize),
    height: Math.max(contentMetrics.contentHeight, textSettings.fontSize),
  }));
  const dragStateRef = useRef<{
    pointerId: number;
    startPointer: Coordinate;
    startAnchor: Coordinate;
  } | null>(null);
  const rotateStateRef = useRef<{
    pointerId: number;
    startAngle: number;
    startRotation: number;
    center: Coordinate;
  } | null>(null);
  const [isCtrlPressed, setIsCtrlPressed] = useState(false);

  const isMoveModifierPressed = useCallback((event: { ctrlKey: boolean; metaKey: boolean }) => {
    return event.ctrlKey || event.metaKey;
  }, []);

  const syncModifierState = useCallback(
    (event: { ctrlKey: boolean; metaKey: boolean }) => {
      const modifierPressed = isMoveModifierPressed(event);
      if (modifierPressed !== isCtrlPressed) {
        setIsCtrlPressed(modifierPressed);
      }
      return modifierPressed;
    },
    [isCtrlPressed, isMoveModifierPressed]
  );

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

  const setContainerRef = useCallback((node: HTMLDivElement | null) => {
    containerRef.current = node;
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
      lastObservedSizeRef.current = null;
      lastMeasuredSizeRef.current = null;
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

  const updateMeasuredSize = useCallback(
    (width: number, height: number) => {
      const nextWidth = Math.max(width, textSettings.fontSize);
      const nextHeight = Math.max(height, textSettings.fontSize);
      const last = lastMeasuredSizeRef.current;
      if (last && Math.abs(last.width - nextWidth) < 0.5 && Math.abs(last.height - nextHeight) < 0.5) {
        return;
      }
      const next = { width: nextWidth, height: nextHeight };
      lastMeasuredSizeRef.current = next;
      setMeasuredSize(next);
      canvasManager.tool.tools.text.updateSessionSize(sessionId, next);
    },
    [canvasManager.tool.tools.text, sessionId, textSettings.fontSize]
  );

  const measureContainer = useCallback(() => {
    const observed = lastObservedSizeRef.current;
    if (observed) {
      updateMeasuredSize(observed.width, observed.height);
      return;
    }
    const node = containerRef.current;
    if (!node) {
      return;
    }
    const width = node.offsetWidth;
    const height = node.offsetHeight;
    if (!width || !height) {
      return;
    }
    updateMeasuredSize(width, height);
  }, [updateMeasuredSize]);

  const handleInput = useCallback(() => {
    const value = (editorRef.current?.innerText ?? '').replace(/\r/g, '');
    setTextValue(value);
    canvasManager.tool.tools.text.updateSessionText(sessionId, value);
  }, [canvasManager.tool.tools.text, sessionId]);

  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      const nativeEvent = event.nativeEvent;
      const isModifierOnlyKey = event.key === 'Control' || event.key === 'Meta';
      if (!isModifierOnlyKey && !isAllowedTextShortcut(nativeEvent)) {
        event.stopPropagation();
        nativeEvent.stopPropagation();
        nativeEvent.stopImmediatePropagation?.();
      }

      if (event.key === 'Enter' && !event.shiftKey && !isComposing) {
        event.preventDefault();
        measureContainer();
        canvasManager.tool.tools.text.requestCommit(sessionId);
      }

      if (event.key === 'Escape') {
        event.preventDefault();
        canvasManager.tool.tools.text.clearSession();
      }
    },
    [canvasManager.tool.tools.text, isComposing, measureContainer, sessionId]
  );

  const handlePaste = useCallback((event: ReactClipboardEvent<HTMLDivElement>) => {
    event.preventDefault();
    const text = event.clipboardData.getData('text/plain');
    document.execCommand('insertText', false, text);
  }, []);

  const handleCompositionStart = useCallback(() => setIsComposing(true), []);
  const handleCompositionEnd = useCallback(() => setIsComposing(false), []);

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

  const fallbackWidth = Math.max(textContainerData.width, textSettings.fontSize);
  const fallbackHeight = Math.max(textContainerData.height, textSettings.fontSize);
  const effectiveWidth = lastMeasuredSizeRef.current ? measuredSize.width : fallbackWidth;
  const effectiveHeight = lastMeasuredSizeRef.current ? measuredSize.height : fallbackHeight;

  useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      if (measureRafIdRef.current !== null) {
        cancelAnimationFrame(measureRafIdRef.current);
      }
      measureRafIdRef.current = requestAnimationFrame(() => {
        measureRafIdRef.current = null;
        const entry = entries[0];
        if (entry) {
          const borderSize = entry.borderBoxSize?.[0];
          const width = borderSize?.inlineSize ?? entry.contentRect.width;
          const height = borderSize?.blockSize ?? entry.contentRect.height;
          if (width && height) {
            lastObservedSizeRef.current = { width, height };
            updateMeasuredSize(width, height);
            return;
          }
        }
        measureContainer();
      });
    });
    observer.observe(node, { box: 'border-box' });
    measureRafIdRef.current = requestAnimationFrame(() => {
      measureRafIdRef.current = null;
      measureContainer();
    });
    return () => {
      observer.disconnect();
      if (measureRafIdRef.current !== null) {
        cancelAnimationFrame(measureRafIdRef.current);
        measureRafIdRef.current = null;
      }
    };
  }, [measureContainer, updateMeasuredSize]);

  const handleContainerPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const modifierPressed = syncModifierState(event);
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
      setIsDragging(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [anchor, canvasManager.tool.$tool, getStagePoint, syncModifierState]
  );

  const handleBorderPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
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
      setIsDragging(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [anchor, getStagePoint]
  );

  const handleContainerPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      syncModifierState(event);
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
    [canvasManager.tool.tools.text, getStagePoint, sessionId, syncModifierState]
  );

  const handleContainerPointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) {
      return;
    }
    event.currentTarget.releasePointerCapture(event.pointerId);
    dragStateRef.current = null;
    setIsDragging(false);
  }, []);

  const handleRotationPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const center = {
        x: textContainerData.x + effectiveWidth / 2,
        y: textContainerData.y + effectiveHeight / 2,
      };
      const startPointer = getStagePoint(event);
      const startAngle = Math.atan2(startPointer.y - center.y, startPointer.x - center.x);
      rotateStateRef.current = {
        pointerId: event.pointerId,
        startAngle,
        startRotation: rotation,
        center,
      };
      setIsRotating(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [effectiveHeight, effectiveWidth, getStagePoint, rotation, textContainerData.x, textContainerData.y]
  );

  const handleRotationPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const rotateState = rotateStateRef.current;
      if (!rotateState || rotateState.pointerId !== event.pointerId) {
        return;
      }
      event.preventDefault();
      const currentPointer = getStagePoint(event);
      const currentAngle = Math.atan2(currentPointer.y - rotateState.center.y, currentPointer.x - rotateState.center.x);
      let nextRotation = rotateState.startRotation + (currentAngle - rotateState.startAngle);
      if (event.shiftKey) {
        const snap = Math.PI / 12;
        nextRotation = Math.round(nextRotation / snap) * snap;
      }
      canvasManager.tool.tools.text.updateSessionRotation(sessionId, nextRotation);
    },
    [canvasManager.tool.tools.text, getStagePoint, sessionId]
  );

  const handleRotationPointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const rotateState = rotateStateRef.current;
    if (!rotateState || rotateState.pointerId !== event.pointerId) {
      return;
    }
    event.currentTarget.releasePointerCapture(event.pointerId);
    rotateStateRef.current = null;
    setIsRotating(false);
  }, []);

  useEffect(() => {
    const handleWindowPointerEnd = () => {
      if (dragStateRef.current) {
        dragStateRef.current = null;
        setIsDragging(false);
      }
      if (rotateStateRef.current) {
        rotateStateRef.current = null;
        setIsRotating(false);
      }
    };
    window.addEventListener('pointerup', handleWindowPointerEnd);
    window.addEventListener('pointercancel', handleWindowPointerEnd);
    return () => {
      window.removeEventListener('pointerup', handleWindowPointerEnd);
      window.removeEventListener('pointercancel', handleWindowPointerEnd);
    };
  }, []);

  useEffect(() => {
    canvasManager.tool.tools.text.updateSessionPosition(sessionId, {
      x: textContainerData.x,
      y: textContainerData.y,
    });
  }, [canvasManager.tool.tools.text, sessionId, textContainerData.x, textContainerData.y]);

  const containerStyle = useMemo(() => {
    return {
      left: `${textContainerData.x}px`,
      top: `${textContainerData.y}px`,
      paddingTop: `${textContainerData.padding}px`,
      paddingBottom: `${textContainerData.padding}px`,
      paddingLeft: `${textContainerData.padding + textContainerData.extraLeftPadding}px`,
      paddingRight: `${textContainerData.padding + textContainerData.extraRightPadding}px`,
      width: `${Math.max(textContainerData.width, textSettings.fontSize)}px`,
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

  const stageScale = stageAttrs.scale || 1;
  const outlineScale = stageScale ? 1 / stageScale : 1;
  const outlineWidthPx = effectiveWidth * stageScale;
  const outlineHeightPx = effectiveHeight * stageScale;
  const borderHitInsidePx = 8;
  const borderHitOutsetPx = 4;
  const borderHitThicknessPx = `${(borderHitInsidePx + borderHitOutsetPx) * stageScale}px`;
  const borderHitOutsetOffsetPx = `${borderHitOutsetPx * stageScale}px`;

  return (
    <Box
      ref={setContainerRef}
      position="absolute"
      pointerEvents="auto"
      borderWidth="0px"
      boxSizing="border-box"
      transform={`rotate(${rotation}rad)`}
      transformOrigin="center"
      sx={{ cursor: isDragging ? 'grabbing' : isCtrlPressed ? 'grab' : 'text' }}
      onPointerDown={handleContainerPointerDown}
      onPointerMove={handleContainerPointerMove}
      onPointerUp={handleContainerPointerUp}
      onPointerCancel={handleContainerPointerUp}
      {...containerStyle}
    >
      <Box
        position="absolute"
        top={0}
        left={0}
        width={`${outlineWidthPx}px`}
        height={`${outlineHeightPx}px`}
        transform={`scale(${outlineScale})`}
        transformOrigin="top left"
        pointerEvents="none"
        borderWidth="2px"
        borderStyle="dashed"
        borderColor="invokeBlue.300"
        borderRadius="md"
      >
        <Box position="absolute" inset={0} pointerEvents="none" borderRadius="md">
          <Box
            position="absolute"
            top={`-${borderHitOutsetOffsetPx}`}
            left={`-${borderHitOutsetOffsetPx}`}
            right={`-${borderHitOutsetOffsetPx}`}
            height={borderHitThicknessPx}
            pointerEvents="auto"
            cursor={isDragging ? 'grabbing' : 'grab'}
            onPointerDown={handleBorderPointerDown}
            onPointerMove={handleContainerPointerMove}
            onPointerUp={handleContainerPointerUp}
            onPointerCancel={handleContainerPointerUp}
          />
          <Box
            position="absolute"
            bottom={`-${borderHitOutsetOffsetPx}`}
            left={`-${borderHitOutsetOffsetPx}`}
            right={`-${borderHitOutsetOffsetPx}`}
            height={borderHitThicknessPx}
            pointerEvents="auto"
            cursor={isDragging ? 'grabbing' : 'grab'}
            onPointerDown={handleBorderPointerDown}
            onPointerMove={handleContainerPointerMove}
            onPointerUp={handleContainerPointerUp}
            onPointerCancel={handleContainerPointerUp}
          />
          <Box
            position="absolute"
            top={`-${borderHitOutsetOffsetPx}`}
            bottom={`-${borderHitOutsetOffsetPx}`}
            left={`-${borderHitOutsetOffsetPx}`}
            width={borderHitThicknessPx}
            pointerEvents="auto"
            cursor={isDragging ? 'grabbing' : 'grab'}
            onPointerDown={handleBorderPointerDown}
            onPointerMove={handleContainerPointerMove}
            onPointerUp={handleContainerPointerUp}
            onPointerCancel={handleContainerPointerUp}
          />
          <Box
            position="absolute"
            top={`-${borderHitOutsetOffsetPx}`}
            bottom={`-${borderHitOutsetOffsetPx}`}
            right={`-${borderHitOutsetOffsetPx}`}
            width={borderHitThicknessPx}
            pointerEvents="auto"
            cursor={isDragging ? 'grabbing' : 'grab'}
            onPointerDown={handleBorderPointerDown}
            onPointerMove={handleContainerPointerMove}
            onPointerUp={handleContainerPointerUp}
            onPointerCancel={handleContainerPointerUp}
          />
        </Box>
        <Box
          position="absolute"
          top={-ROTATE_ANCHOR_LINE_LENGTH}
          left="50%"
          transform="translateX(-50%)"
          width="0"
          height={`${ROTATE_ANCHOR_LINE_LENGTH}px`}
          borderLeftWidth="2px"
          borderLeftStyle="dashed"
          borderLeftColor={ROTATE_ANCHOR_STROKE}
        />
        <Box
          position="absolute"
          top={-(ROTATE_ANCHOR_LINE_LENGTH + ROTATE_ANCHOR_BASE_SIZE)}
          left="50%"
          transform="translateX(-50%)"
          width={`${ROTATE_ANCHOR_SIZE}px`}
          height={`${ROTATE_ANCHOR_SIZE}px`}
          borderRadius="full"
          bg={ROTATE_ANCHOR_FILL}
          borderWidth="2px"
          borderColor={ROTATE_ANCHOR_STROKE}
          cursor={isRotating ? 'grabbing' : 'grab'}
          pointerEvents="auto"
          onPointerDown={handleRotationPointerDown}
          onPointerMove={handleRotationPointerMove}
          onPointerUp={handleRotationPointerUp}
          onPointerCancel={handleRotationPointerUp}
        />
      </Box>
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
          cursor: isCtrlPressed ? 'grab' : 'text',
          display: 'inline-block',
          pointerEvents: isCtrlPressed ? 'none' : 'auto',
          ...textStyle,
        }}
      />
    </Box>
  );
};
