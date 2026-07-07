import { Box, Button, ButtonGroup, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import type { RectCorner } from 'features/controlLayers/components/SplatOverlay/rectTransforms';
import { moveRect, resizeRectFromCorner } from 'features/controlLayers/components/SplatOverlay/rectTransforms';
import type { SplatScene } from 'features/controlLayers/components/SplatOverlay/splatScene';
import type { SplatRect } from 'features/controlLayers/components/SplatOverlay/state';
import {
  $splatOverlay,
  clearSplatOverlay,
  updateSplatOverlayRect,
} from 'features/controlLayers/components/SplatOverlay/state';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { lazy, memo, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('canvas');

// Lazy so the three.js + Spark chunk is only fetched when the overlay opens (and a load failure there
// can't blank the rest of the app). `type`-only import of SplatScene above is erased, so it stays lazy.
const SplatViewer = lazy(() => import('features/controlLayers/components/SplatOverlay/SplatViewer'));

// Chrome sizes in screen px (divided by stage scale so they're constant at any zoom).
const OUTLINE_WIDTH_PX = 2;
const EDGE_HIT_PX = 10;
const HANDLE_SIZE_PX = 12;
const MIN_RECT_SIZE = 16; // world px

const CORNERS: { corner: RectCorner; cursor: string }[] = [
  { corner: 'nw', cursor: 'nwse-resize' },
  { corner: 'ne', cursor: 'nesw-resize' },
  { corner: 'sw', cursor: 'nesw-resize' },
  { corner: 'se', cursor: 'nwse-resize' },
];

type RotateMode = 'orbit' | 'object';

const CONTROL_HINTS: Record<RotateMode, string[]> = {
  orbit: ['Drag to orbit view', 'Right-drag to pan', 'Scroll to zoom', 'Edges to move', 'Corners to resize'],
  object: ['Drag to rotate object', 'Right-drag to pan', 'Scroll to zoom', 'Edges to move', 'Corners to resize'],
};

type DragState = {
  pointerId: number;
  mode: 'move' | RectCorner;
  startPointer: { x: number; y: number };
  startRect: SplatRect;
};

/**
 * The in-canvas 3D (Gaussian-splat) overlay: a transparent three.js viewport pinned to a world-space
 * footprint rect, so the splat composites live over the actual canvas content. Dragging inside the frame
 * either orbits the view or rotates the object itself (any axis, including roll) — a toolbar toggle picks
 * the mode. Drag the frame edges to move it; drag the corners to resize (shift = keep aspect). The stage
 * stays interactive outside the frame, so you can pan/zoom the canvas for precise placement. Commit bakes
 * the current framing to a new raster layer at the current rect.
 */
export const CanvasSplatOverlay = memo(() => {
  const canvasManager = useCanvasManager();
  const state = useStore($splatOverlay);
  const stageAttrs = useStore(canvasManager.stage.$stageAttrs);
  const dispatch = useAppDispatch();
  const sceneRef = useRef<SplatScene | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [isCommitting, setIsCommitting] = useState(false);
  const [rotateMode, setRotateMode] = useState<RotateMode>('orbit');
  // The scene remounts on asset/session changes, so the current mode is kept in a ref and re-applied
  // whenever a scene arrives.
  const rotateModeRef = useRef<RotateMode>(rotateMode);

  const onSceneReady = useCallback((scene: SplatScene | null) => {
    sceneRef.current = scene;
    scene?.setRotateObjectMode(rotateModeRef.current === 'object');
  }, []);

  const applyRotateMode = useCallback((mode: RotateMode) => {
    rotateModeRef.current = mode;
    setRotateMode(mode);
    sceneRef.current?.setRotateObjectMode(mode === 'object');
  }, []);
  const onOrbitMode = useCallback(() => applyRotateMode('orbit'), [applyRotateMode]);
  const onObjectMode = useCallback(() => applyRotateMode('object'), [applyRotateMode]);

  const getWorldPoint = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const containerRect = canvasManager.stage.container.getBoundingClientRect();
      return {
        x: (e.clientX - containerRect.left - stageAttrs.x) / stageAttrs.scale,
        y: (e.clientY - containerRect.top - stageAttrs.y) / stageAttrs.scale,
      };
    },
    [canvasManager.stage.container, stageAttrs.x, stageAttrs.y, stageAttrs.scale]
  );

  const startDrag = useCallback(
    (e: ReactPointerEvent<HTMLElement>, mode: DragState['mode']) => {
      const current = $splatOverlay.get();
      if (e.button !== 0 || !current) {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      dragRef.current = {
        pointerId: e.pointerId,
        mode,
        startPointer: getWorldPoint(e),
        startRect: current.rect,
      };
      e.currentTarget.setPointerCapture(e.pointerId);
    },
    [getWorldPoint]
  );

  const onDragMove = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const drag = dragRef.current;
      if (!drag || drag.pointerId !== e.pointerId) {
        return;
      }
      e.preventDefault();
      const point = getWorldPoint(e);
      const dx = point.x - drag.startPointer.x;
      const dy = point.y - drag.startPointer.y;
      const next =
        drag.mode === 'move'
          ? moveRect(drag.startRect, dx, dy)
          : resizeRectFromCorner(drag.startRect, drag.mode, dx, dy, {
              keepAspect: e.shiftKey,
              minSize: MIN_RECT_SIZE,
            });
      updateSplatOverlayRect(next);
    },
    [getWorldPoint]
  );

  const endDrag = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== e.pointerId) {
      return;
    }
    e.currentTarget.releasePointerCapture(e.pointerId);
    dragRef.current = null;
  }, []);

  const handleCommit = useCallback(() => {
    const scene = sceneRef.current;
    const current = $splatOverlay.get();
    if (!scene || current?.status !== 'ready') {
      return;
    }
    const { rect } = current;
    setIsCommitting(true);
    void (async () => {
      try {
        // Render at the footprint the user framed, so the baked layer lands at the right size + position.
        const blob = await scene.capture(Math.max(1, Math.round(rect.width)), Math.max(1, Math.round(rect.height)));
        if (!blob) {
          throw new Error('Failed to capture splat');
        }
        const file = new File([blob], 'splat.png', { type: 'image/png' });
        const imageDTO = await uploadImage({ file, image_category: 'general', is_intermediate: true, silent: true });
        dispatch(
          rasterLayerAdded({
            overrides: {
              objects: [imageDTOToImageObject(imageDTO)],
              position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
            },
            isSelected: true,
          })
        );
        clearSplatOverlay();
      } catch (error) {
        log.error({ error: String(error) }, 'Failed to commit 3D layer to canvas');
        setIsCommitting(false);
      }
    })();
  }, [dispatch]);

  // This component stays mounted across open/close (it just renders null when closed), so per-session UI
  // state would persist between sessions. Reset it whenever we're not in an active "ready" session — i.e.
  // after a successful commit (status -> null) or when a new conversion starts (status -> loading).
  const status = state?.status;
  useEffect(() => {
    if (status !== 'ready') {
      setIsCommitting(false);
      rotateModeRef.current = 'orbit';
      setRotateMode('orbit');
    }
  }, [status]);

  // Escape cancels the session (capture phase so canvas hotkeys don't also react).
  const isOpen = state !== null;
  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        clearSplatOverlay();
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => {
      window.removeEventListener('keydown', onKeyDown, true);
    };
  }, [isOpen]);

  if (!state) {
    return null;
  }

  const { rect } = state;
  const scale = stageAttrs.scale || 1;
  // Chrome sized in world units so it renders at a constant screen size regardless of stage zoom.
  const outlineWidth = OUTLINE_WIDTH_PX / scale;
  const edgeHit = EDGE_HIT_PX / scale;
  const edgeHalf = edgeHit / 2;
  const handleSize = HANDLE_SIZE_PX / scale;
  const handleHalf = handleSize / 2;
  const chromeEnabled = !isCommitting;

  return (
    <Flex pointerEvents="none" position="absolute" inset={0} sx={{ isolation: 'isolate' }}>
      {/* World-pinned layer: children are positioned in canvas/world coordinates. */}
      <Box
        pointerEvents="none"
        position="absolute"
        inset={0}
        transform={`translate(${stageAttrs.x}px, ${stageAttrs.y}px) scale(${scale})`}
        transformOrigin="top left"
      >
        <Box
          position="absolute"
          pointerEvents="none"
          style={{ left: rect.x, top: rect.y, width: rect.width, height: rect.height }}
        >
          {state.status === 'ready' && (
            <Box position="absolute" inset={0} pointerEvents={chromeEnabled ? 'auto' : 'none'}>
              <Suspense fallback={null}>
                <SplatViewer assetUrl={state.assetUrl} stageScale={scale} onSceneReady={onSceneReady} />
              </Suspense>
            </Box>
          )}
          {state.status === 'loading' && (
            <Flex position="absolute" inset={0} alignItems="center" justifyContent="center" pointerEvents="none">
              <Spinner size="lg" />
            </Flex>
          )}
          {/* Dashed footprint outline */}
          <Box
            position="absolute"
            inset={0}
            pointerEvents="none"
            borderStyle="dashed"
            borderColor="invokeBlue.300"
            style={{ borderWidth: outlineWidth }}
          />
          {/* Edge strips: drag to move the footprint. Rendered after the viewer so they win near the border. */}
          {chromeEnabled &&
            (
              [
                { key: 'n', style: { top: -edgeHalf, left: edgeHalf, right: edgeHalf, height: edgeHit } },
                { key: 's', style: { bottom: -edgeHalf, left: edgeHalf, right: edgeHalf, height: edgeHit } },
                { key: 'w', style: { left: -edgeHalf, top: edgeHalf, bottom: edgeHalf, width: edgeHit } },
                { key: 'e', style: { right: -edgeHalf, top: edgeHalf, bottom: edgeHalf, width: edgeHit } },
              ] as const
            ).map(({ key, style }) => (
              <Box
                key={key}
                position="absolute"
                pointerEvents="auto"
                cursor="move"
                style={style}
                onPointerDown={(e) => startDrag(e, 'move')}
                onPointerMove={onDragMove}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
              />
            ))}
          {/* Corner handles: drag to resize (shift = keep aspect). */}
          {chromeEnabled &&
            CORNERS.map(({ corner, cursor }) => (
              <Box
                key={corner}
                position="absolute"
                pointerEvents="auto"
                bg="invokeBlue.50"
                borderColor="invokeBlue.500"
                borderRadius="1px"
                style={{
                  width: handleSize,
                  height: handleSize,
                  borderWidth: outlineWidth,
                  cursor,
                  top: corner === 'nw' || corner === 'ne' ? -handleHalf : undefined,
                  bottom: corner === 'sw' || corner === 'se' ? -handleHalf : undefined,
                  left: corner === 'nw' || corner === 'sw' ? -handleHalf : undefined,
                  right: corner === 'ne' || corner === 'se' ? -handleHalf : undefined,
                }}
                onPointerDown={(e) => startDrag(e, corner)}
                onPointerMove={onDragMove}
                onPointerUp={endDrag}
                onPointerCancel={endDrag}
              />
            ))}
        </Box>
      </Box>
      {/* Screen-space toolbar: always visible regardless of pan/zoom. Wraps on narrow panels — the hint
          breaks between segments and the button pair drops to its own row rather than clipping. */}
      <Flex
        position="absolute"
        top={2}
        insetInlineStart="50%"
        transform="translateX(-50%)"
        columnGap={2}
        rowGap={1}
        alignItems="center"
        justifyContent="center"
        flexWrap="wrap"
        maxW="calc(100% - 16px)"
        pointerEvents="auto"
        bg="base.800"
        borderRadius="base"
        px={2}
        py={1.5}
        shadow="dark-lg"
      >
        {state.status === 'loading' ? (
          <Flex gap={2} alignItems="center" px={1}>
            <Spinner size="sm" />
            <Text fontSize="sm" whiteSpace="nowrap">
              Generating 3D…
            </Text>
          </Flex>
        ) : (
          <>
            <ButtonGroup isAttached size="sm" flexShrink={0}>
              <Button colorScheme={rotateMode === 'orbit' ? 'invokeBlue' : 'base'} onClick={onOrbitMode}>
                Orbit view
              </Button>
              <Button colorScheme={rotateMode === 'object' ? 'invokeBlue' : 'base'} onClick={onObjectMode}>
                Rotate object
              </Button>
            </ButtonGroup>
            <Flex px={1} columnGap={1.5} flexWrap="wrap" justifyContent="center">
              {CONTROL_HINTS[rotateMode].map((hint, i) => (
                <Text key={hint} fontSize="xs" color="base.300" whiteSpace="nowrap">
                  {hint}
                  {i < CONTROL_HINTS[rotateMode].length - 1 ? ' ·' : ''}
                </Text>
              ))}
            </Flex>
          </>
        )}
        <Flex gap={2} flexShrink={0}>
          <Button size="sm" onClick={clearSplatOverlay} isDisabled={isCommitting}>
            Cancel
          </Button>
          <Button
            size="sm"
            colorScheme="invokeYellow"
            onClick={handleCommit}
            isLoading={isCommitting}
            isDisabled={state.status !== 'ready'}
          >
            Commit to layer
          </Button>
        </Flex>
      </Flex>
    </Flex>
  );
});
CanvasSplatOverlay.displayName = 'CanvasSplatOverlay';
