import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  ControlLayerViewMode,
} from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Box, Flex, HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { useDndContext } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { getDocumentNode, isControlMapCurrent, lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import {
  getControlSourceSnapshot,
  subscribeControlSourceSnapshot,
  type ControlSourceSnapshot,
} from '@workbench/canvas-operations/api';
import { getControlImageState } from '@workbench/canvasLayerContent';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ImagePlusIcon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { CanvasFloatingBar } from './CanvasFloatingBar';
import {
  useCanvasActiveTool,
  useCanvasDocumentEditingLocked,
  useCanvasEngineRead,
  useControlDetectionStatus,
  useControlLayerView,
} from './engineStoreHooks';

type OverlayEngine = Pick<CanvasEngineHandle, 'exports' | 'interaction' | 'previews'>;

// The document index hands back the same node until it changes, so other layers' edits do not re-render this.
const selectSelectedControlLayer = (project: {
  canvas: { document: Pick<CanvasDocumentContractV3, 'stacks' | 'selectedLayerId'> };
}): CanvasControlLayerContract | null => {
  const node = getDocumentNode(project.canvas.document, project.canvas.document.selectedLayerId);
  return node?.type === 'control' ? node : null;
};

// Whether the selected layer draws at all: enabled and shown, with every ancestor.
const selectSelectedLayerDisplayed = (project: { canvas: { document: CanvasDocumentContractV3 } }): boolean => {
  const { document } = project.canvas;
  const leaf = document.selectedLayerId ? lookupDocumentLeaf(document, document.selectedLayerId) : null;
  return !!leaf && leaf.contributionEnabled && !leaf.documentHidden;
};

// Below the center view's header row, as the tool strip sits.
const TOP_SLOT_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

/**
 * Canvas chrome for the selected control layer: a hint over an empty layer pointing at the image sources, a
 * top-center chip while its map is detected in the background, and in its place once the map is current, an
 * Original / Map / Overlay switch comparing the map with the picture it was detected from.
 */
export const ControlLayerCanvasOverlay = ({ engine }: { engine: OverlayEngine }) => {
  const layer = useActiveProjectSelector(selectSelectedControlLayer);
  return layer ? <SelectedControlLayerOverlay engine={engine} layer={layer} /> : null;
};

const SelectedControlLayerOverlay = ({
  engine,
  layer,
}: {
  engine: OverlayEngine;
  layer: CanvasControlLayerContract;
}) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const status = useControlDetectionStatus(engine, layer.id);
  const { active } = useDndContext();
  const subscribeSource = useCallback(
    (listener: () => void) => subscribeControlSourceSnapshot(engine, listener),
    [engine]
  );
  const getSourceSnapshot = useCallback(() => getControlSourceSnapshot(engine), [engine]);
  const isUploading = useExternalStoreSelector(
    subscribeSource,
    getSourceSnapshot,
    useCallback((snapshot: ControlSourceSnapshot) => snapshot.uploading.has(layer.id), [layer.id])
  );
  const { notifications } = useWorkbenchCommands();
  const isDocumentEditingLocked = useCanvasDocumentEditingLocked(engine);
  const isDisplayed = useActiveProjectSelector(selectSelectedLayerDisplayed);
  const controlView = useControlLayerView(engine);
  // The requested view shows at once; the engine's view takes over when the request settles.
  const [pendingView, setPendingView] = useState<{ layerId: string; mode: ControlLayerViewMode } | null>(null);
  const viewOptions = useMemo(
    () => [
      { label: t('widgets.canvas.controlView.original'), value: 'original' },
      { label: t('widgets.canvas.controlView.map'), value: 'map' },
      { label: t('widgets.canvas.controlView.overlay'), value: 'overlay' },
    ],
    [t]
  );
  const showView = useCallback(
    (value: string) => {
      const request = { layerId: layer.id, mode: value as ControlLayerViewMode };
      setPendingView(request);
      void engine.previews.setControlLayerView(request.layerId, request.mode).then((outcome) => {
        setPendingView((current) => (current === request ? null : current));
        // Other refusals explain themselves: the switch is hidden, or the engine's view already reflects them.
        if (outcome === 'failed' || outcome === 'over-budget') {
          notifications.add({
            kind: 'error',
            title: t(
              outcome === 'failed' ? 'widgets.canvas.controlView.failed' : 'widgets.canvas.controlView.overBudget'
            ),
          });
        }
      });
    },
    [engine, layer.id, notifications, t]
  );
  const shownView =
    pendingView?.layerId === layer.id ? pendingView.mode : controlView?.layerId === layer.id ? controlView.view : 'map';
  const persistedState = getControlImageState(layer, false);
  // Only a source whose strokes may be unsaved asks the engine cache; every other layer reads a constant.
  const imageState = useCanvasEngineRead(engine, () =>
    persistedState === 'empty' && !layer.original
      ? getControlImageState(layer, engine.exports.hasExportableLayerContent(layer.id))
      : persistedState
  );
  const isGalleryDrag = isGalleryImageDragData(active?.data.current);
  // Draw hides the hint: the user is about to give the layer its pixels by hand. A gallery drag shows the drop zones
  // in its place. An upload on its way (from Properties or a dropped file) keeps the card up, saying so.
  const isHintVisible =
    imageState === 'empty' && (isUploading || (!isGalleryDrag && activeTool !== 'brush' && activeTool !== 'eraser'));
  // Idle status keeps the switch and the detection chip apart: one or the other occupies the slot. The engine refuses
  // a view while an operation holds the document, and a layer that does not draw has nothing to compare.
  const isViewSwitchVisible =
    layer.original !== undefined &&
    !!layer.detection &&
    isControlMapCurrent(layer) &&
    isDisplayed &&
    status.status === 'idle' &&
    !isDocumentEditingLocked &&
    !isGalleryDrag;
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
          <CanvasFloatingBar aria-busy={isUploading} maxW="72" pointerEvents="none" px="3" py="2" role="note">
            <HStack align="center" gap="2.5">
              {isUploading ? (
                <Spinner boxSize="4" color="fg.muted" flexShrink={0} />
              ) : (
                <Icon as={ImagePlusIcon} boxSize="5" color="fg.muted" flexShrink={0} />
              )}
              <Stack gap="0.5" minW="0">
                <Text fontSize="md" fontWeight="600">
                  {t(isUploading ? 'widgets.canvas.controlHint.uploading' : 'widgets.canvas.controlHint.title')}
                </Text>
                {isUploading ? null : (
                  <Text color="fg.muted" fontSize="md">
                    {t('widgets.canvas.controlHint.body')}
                  </Text>
                )}
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
                    : t(
                        status.status === 'queued'
                          ? 'widgets.layers.control.detectionQueued'
                          : 'widgets.layers.control.detecting',
                        { detector: detectorLabel }
                      )}
                </Text>
              </HStack>
            </CanvasFloatingBar>
          ) : null}
        </Box>
        {isViewSwitchVisible ? (
          <CanvasFloatingBar>
            <SegmentedControl
              ariaLabel={t('widgets.canvas.controlView.label')}
              isFullWidth={false}
              options={viewOptions}
              value={shownView}
              onChange={showView}
            />
          </CanvasFloatingBar>
        ) : null}
      </Flex>
    </>
  );
};
