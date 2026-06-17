import { Button, Flex, Spinner, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import type { SplatScene } from 'features/controlLayers/components/SplatOverlay/splatScene';
import { $splatOverlay, clearSplatOverlay } from 'features/controlLayers/components/SplatOverlay/state';
import { rasterLayerAdded } from 'features/controlLayers/store/canvasSlice';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { lazy, memo, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('canvas');

// Lazy so the three.js + Spark chunk is only fetched when the overlay opens (and a load failure there
// can't blank the rest of the app). `type`-only import of SplatScene above is erased, so it stays lazy.
const SplatViewer = lazy(() => import('features/controlLayers/components/SplatOverlay/SplatViewer'));

export const CanvasSplatOverlay = memo(() => {
  const state = useStore($splatOverlay);
  const dispatch = useAppDispatch();
  const sceneRef = useRef<SplatScene | null>(null);
  const [isCommitting, setIsCommitting] = useState(false);

  const onSceneReady = useCallback((scene: SplatScene | null) => {
    sceneRef.current = scene;
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
        // Render at the source layer's footprint so the baked layer lands at the right size + position.
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

  // This component stays mounted across open/close (it just renders null when closed), so `isCommitting`
  // would persist between sessions. Reset it whenever we're not in an active "ready" session — i.e. after
  // a successful commit (status -> null) or when a new conversion starts (status -> loading).
  const status = state?.status;
  useEffect(() => {
    if (status !== 'ready') {
      setIsCommitting(false);
    }
  }, [status]);

  if (!state) {
    return null;
  }

  const aspect = state.rect.height > 0 ? state.rect.width / state.rect.height : 1;

  return (
    <Flex position="absolute" inset={0} zIndex={1} bg="blackAlpha.600" alignItems="center" justifyContent="center">
      {state.status === 'ready' && (
        <Suspense fallback={<Spinner size="xl" />}>
          <SplatViewer assetUrl={state.assetUrl} aspect={aspect} onSceneReady={onSceneReady} />
        </Suspense>
      )}
      {state.status === 'loading' && (
        <Flex flexDir="column" alignItems="center" gap={3} pointerEvents="none">
          <Spinner size="xl" />
          <Text>Generating 3D…</Text>
        </Flex>
      )}
      <Flex position="absolute" top={2} insetInlineStart="50%" transform="translateX(-50%)" gap={2}>
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
  );
});
CanvasSplatOverlay.displayName = 'CanvasSplatOverlay';
