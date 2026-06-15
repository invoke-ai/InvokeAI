import type { KeyboardEvent } from 'react';

import { Box, Flex } from '@chakra-ui/react';
import { useActiveProject, useWorkbenchDispatch } from '@workbench/WorkbenchContext';

import { CanvasDocumentFrame, CanvasPlaneImage, EmptyCanvasFrame, ToolScrubber } from './CanvasDocumentFrame';
import { CanvasStagingControls, EmptyStagingControls } from './CanvasStagingControls';

export const CanvasWidgetView = () => {
  const activeProject = useActiveProject();
  const dispatch = useWorkbenchDispatch();
  const { document, stagingArea } = activeProject.canvas;
  const { layers } = document;
  const selectedCandidate = stagingArea.pendingImages[stagingArea.selectedImageIndex];
  const selectedImage = stagingArea.isVisible ? selectedCandidate : undefined;
  const hasStagedCandidates = stagingArea.pendingImages.length > 0;
  const hasMultipleCandidates = stagingArea.pendingImages.length > 1;
  const renderedLayers = [...layers].reverse();
  const hasCanvasContent = renderedLayers.length > 0 || Boolean(selectedImage);
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!hasStagedCandidates) {
      return;
    }

    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      dispatch({ direction: -1, type: 'cycleStagedImage' });
      return;
    }

    if (event.key === 'ArrowRight') {
      event.preventDefault();
      dispatch({ direction: 1, type: 'cycleStagedImage' });
      return;
    }

    if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault();
      dispatch({ type: 'discardSelectedStagedImage' });
    }
  };

  return (
    <Box
      aria-label="Canvas surface"
      bg="bg.inset"
      h="full"
      overflow="hidden"
      position="relative"
      tabIndex={0}
      w="full"
      bgImage="radial-gradient({colors.fg.grid} 1.5px, transparent 1.5px)"
      bgSize="28px 28px"
      onKeyDown={onKeyDown}
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
          onAccept={() => dispatch({ type: 'acceptStagedImage' })}
          onCycle={(direction) => dispatch({ direction, type: 'cycleStagedImage' })}
          onDiscardAll={() => dispatch({ type: 'discardAllStagedImages' })}
          onDiscardSelected={() => dispatch({ type: 'discardSelectedStagedImage' })}
          onSelectImage={(imageIndex) => dispatch({ imageIndex, type: 'setStagedImageIndex' })}
          onToggleThumbnails={() => dispatch({ type: 'toggleCanvasStagingThumbnailsVisibility' })}
          onToggleVisibility={() => dispatch({ type: 'toggleCanvasStagingVisibility' })}
        />
      ) : (
        <EmptyStagingControls />
      )}
    </Box>
  );
};
