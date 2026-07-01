import type { WidgetViewProps } from '@workbench/types';

import { Box, Flex } from '@chakra-ui/react';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useEffectEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { CanvasDocumentFrame, CanvasPlaneImage, EmptyCanvasFrame, ToolScrubber } from './CanvasDocumentFrame';
import { CanvasStagingControls, EmptyStagingControls } from './CanvasStagingControls';

export const CanvasWidgetView = ({ runtime }: WidgetViewProps) => {
  const { t } = useTranslation();
  const canvas = useActiveProjectSelector((project) => project.canvas);
  const dispatch = useWorkbenchDispatch();
  const { document, stagingArea } = canvas;
  const { layers } = document;
  const selectedCandidate = stagingArea.pendingImages[stagingArea.selectedImageIndex];
  const selectedImage = stagingArea.isVisible ? selectedCandidate : undefined;
  const hasStagedCandidates = stagingArea.pendingImages.length > 0;
  const hasMultipleCandidates = stagingArea.pendingImages.length > 1;
  const renderedLayers = [...layers].reverse();
  const hasCanvasContent = renderedLayers.length > 0 || Boolean(selectedImage);

  const executeCanvasHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'canvas.prevEntity' && hasStagedCandidates) {
      dispatch({ direction: -1, type: 'cycleStagedImage' });
    } else if (commandId === 'canvas.nextEntity' && hasStagedCandidates) {
      dispatch({ direction: 1, type: 'cycleStagedImage' });
    } else if (commandId === 'canvas.deleteSelected' && hasStagedCandidates) {
      dispatch({ type: 'discardSelectedStagedImage' });
    } else if (commandId === 'canvas.undo') {
      dispatch({ type: 'undoProjectChange' });
    } else if (commandId === 'canvas.redo') {
      dispatch({ type: 'redoProjectChange' });
    }
  });

  const handleAccept = useCallback(() => dispatch({ type: 'acceptStagedImage' }), [dispatch]);
  const handleCycle = useCallback((direction: -1 | 1) => dispatch({ direction, type: 'cycleStagedImage' }), [dispatch]);
  const handleDiscardAll = useCallback(() => dispatch({ type: 'discardAllStagedImages' }), [dispatch]);
  const handleDiscardSelected = useCallback(() => dispatch({ type: 'discardSelectedStagedImage' }), [dispatch]);
  const handleSelectImage = useCallback(
    (imageIndex: number) => dispatch({ imageIndex, type: 'setStagedImageIndex' }),
    [dispatch]
  );
  const handleToggleThumbnails = useCallback(
    () => dispatch({ type: 'toggleCanvasStagingThumbnailsVisibility' }),
    [dispatch]
  );
  const handleToggleVisibility = useCallback(() => dispatch({ type: 'toggleCanvasStagingVisibility' }), [dispatch]);

  useEffect(() => {
    const hotkeys = [
      ['canvas.prevEntity', t('widgets.canvas.commands.previousEntity'), ['alt+[', 'arrowleft']],
      ['canvas.nextEntity', t('widgets.canvas.commands.nextEntity'), ['alt+]', 'arrowright']],
      ['canvas.deleteSelected', t('widgets.canvas.commands.deleteSelected'), ['delete', 'backspace']],
      ['canvas.undo', t('widgets.canvas.commands.undo'), ['mod+z']],
      ['canvas.redo', t('widgets.canvas.commands.redo'), ['mod+shift+z', 'mod+y']],
    ] as const;
    const disposers = hotkeys.flatMap(([id, title, defaultKeys]) => [
      runtime.commands.register({ handler: () => executeCanvasHotkey(id), id, title }),
      runtime.hotkeys.register({ commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);

  return (
    <Box
      aria-label={t('widgets.canvas.surface')}
      bg="bg.inset"
      h="full"
      overflow="hidden"
      position="relative"
      tabIndex={0}
      w="full"
      bgImage="radial-gradient({colors.fg.grid} 1.5px, transparent 1.5px)"
      bgSize="28px 28px"
    >
      <ToolScrubber />
      <Flex align="center" h="full" justify="center" p="12">
        <CanvasDocumentFrame
          documentHeight={document.height}
          documentWidth={document.width}
          hasContent={hasCanvasContent}
        >
          {hasCanvasContent ? null : <EmptyCanvasFrame />}
          {renderedLayers.map((layer) => (
            <CanvasPlaneImage
              key={layer.id}
              image={layer}
              opacity={selectedImage ? Math.min(layer.placement.opacity, 0.72) : layer.placement.opacity}
              placement={layer.placement}
              planeHeight={document.height}
              planeWidth={document.width}
            />
          ))}
          {selectedImage ? (
            <CanvasPlaneImage
              image={selectedImage}
              isStaged
              opacity={selectedImage.placement.opacity}
              placement={selectedImage.placement}
              planeHeight={document.height}
              planeWidth={document.width}
            />
          ) : null}
        </CanvasDocumentFrame>
      </Flex>
      {hasStagedCandidates && selectedCandidate ? (
        <CanvasStagingControls
          areThumbnailsVisible={stagingArea.areThumbnailsVisible}
          hasMultipleCandidates={hasMultipleCandidates}
          isVisible={stagingArea.isVisible}
          pendingImages={stagingArea.pendingImages}
          selectedCandidate={selectedCandidate}
          selectedImageIndex={stagingArea.selectedImageIndex}
          onAccept={handleAccept}
          onCycle={handleCycle}
          onDiscardAll={handleDiscardAll}
          onDiscardSelected={handleDiscardSelected}
          onSelectImage={handleSelectImage}
          onToggleThumbnails={handleToggleThumbnails}
          onToggleVisibility={handleToggleVisibility}
        />
      ) : (
        <EmptyStagingControls />
      )}
    </Box>
  );
};
