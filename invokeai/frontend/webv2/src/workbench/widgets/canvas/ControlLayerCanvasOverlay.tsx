import type { CanvasControlLayerContract, CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Box, Flex, HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndContext } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { getDocumentNode } from '@workbench/canvas-engine/api';
import { getControlImageState } from '@workbench/canvasLayerContent';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { ImagePlusIcon, TriangleAlertIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { CanvasFloatingBar } from './CanvasFloatingBar';
import { useCanvasActiveTool, useCanvasEngineRead, useControlDetectionStatus } from './engineStoreHooks';

type OverlayEngine = Pick<CanvasEngineHandle, 'exports' | 'interaction'>;

// The document index hands back the same node until it changes, so other layers' edits do not re-render this.
const selectSelectedControlLayer = (project: {
  canvas: { document: Pick<CanvasDocumentContractV3, 'stacks' | 'selectedLayerId'> };
}): CanvasControlLayerContract | null => {
  const node = getDocumentNode(project.canvas.document, project.canvas.document.selectedLayerId);
  return node?.type === 'control' ? node : null;
};

// Below the center view's header row, as the tool strip sits.
const TOP_SLOT_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

/**
 * Canvas chrome for the selected control layer: a hint over an empty layer pointing at the image sources, and a
 * top-center chip while its map is detected in the background.
 */
export const ControlLayerCanvasOverlay = ({
  engine,
  isInteractionLocked,
}: {
  engine: OverlayEngine;
  isInteractionLocked: boolean;
}) => {
  const layer = useActiveProjectSelector(selectSelectedControlLayer);
  return layer ? (
    <SelectedControlLayerOverlay engine={engine} isInteractionLocked={isInteractionLocked} layer={layer} />
  ) : null;
};

const SelectedControlLayerOverlay = ({
  engine,
  isInteractionLocked,
  layer,
}: {
  engine: OverlayEngine;
  isInteractionLocked: boolean;
  layer: CanvasControlLayerContract;
}) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const status = useControlDetectionStatus(engine, layer.id);
  const { active } = useDndContext();
  const persistedState = getControlImageState(layer, false);
  // Only a source whose strokes may be unsaved asks the engine cache; every other layer reads a constant.
  const imageState = useCanvasEngineRead(engine, () =>
    persistedState === 'empty' && !layer.original
      ? getControlImageState(layer, engine.exports.hasExportableLayerContent(layer.id))
      : persistedState
  );
  // Draw hides the hint: the user is about to give the layer its pixels by hand. A locked surface (staging, a running
  // canvas generation) refuses drops, and a gallery drag shows the drop zones in its place.
  const isHintVisible =
    imageState === 'empty' &&
    !isInteractionLocked &&
    !isGalleryImageDragData(active?.data.current) &&
    activeTool !== 'brush' &&
    activeTool !== 'eraser';
  const detector =
    status.status === 'processing'
      ? status.detector
      : status.status === 'idle'
        ? null
        : (layer.detection?.type ?? null);
  const detectorLabel = detector ? t(`widgets.layers.control.filters.${detector}`, detector) : null;

  return (
    <>
      {imageState === 'empty' ? (
        <Flex
          align="center"
          bg="bg.inset/60"
          inset="0"
          justify="center"
          opacity={isHintVisible ? 1 : 0}
          pointerEvents="none"
          position="absolute"
          transitionDuration="var(--wb-motion-duration-medium)"
          transitionProperty="opacity, visibility"
          transitionTimingFunction="ease"
          visibility={isHintVisible ? 'visible' : 'hidden'}
          zIndex="1"
        >
          <CanvasFloatingBar maxW="72" pointerEvents="none" px="3" py="2" role="note">
            <HStack align="center" gap="2.5">
              <Icon as={ImagePlusIcon} boxSize="5" color="fg.muted" flexShrink={0} />
              <Stack gap="0.5" minW="0">
                <Text fontSize="md" fontWeight="600">
                  {t('widgets.canvas.controlHint.title')}
                </Text>
                <Text color="fg.muted" fontSize="md">
                  {t('widgets.canvas.controlHint.body')}
                </Text>
              </Stack>
            </HStack>
          </CanvasFloatingBar>
        </Flex>
      ) : null}
      {/* The top-center slot; the live region stays mounted so a run that starts or fails is announced. */}
      <Flex
        gap="2"
        justify="center"
        left="50%"
        maxW="calc(100% - var(--chakra-spacing-4))"
        pointerEvents="none"
        position="absolute"
        top={TOP_SLOT_TOP}
        transform="translateX(-50%)"
        w="max-content"
        zIndex="2"
      >
        <Box aria-live="polite" role="status">
          {detectorLabel && status.status !== 'idle' ? (
            <CanvasFloatingBar pointerEvents="none" px="2.5" py="1.5">
              <HStack gap="2">
                {status.status === 'error' ? (
                  <Icon as={TriangleAlertIcon} boxSize="3.5" color="fg.warning" flexShrink={0} />
                ) : (
                  <Spinner boxSize="3" flexShrink={0} />
                )}
                <Text fontSize="md">
                  {status.status === 'error'
                    ? t('widgets.layers.control.detectionFailed', {
                        detector: detectorLabel,
                        reason: t(`widgets.layers.control.detectionErrors.${status.error.code}`),
                      })
                    : t('widgets.layers.control.detecting', { detector: detectorLabel })}
                </Text>
              </HStack>
            </CanvasFloatingBar>
          ) : null}
        </Box>
      </Flex>
    </>
  );
};
