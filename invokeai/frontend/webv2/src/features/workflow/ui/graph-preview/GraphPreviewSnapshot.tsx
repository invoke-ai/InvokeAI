import type { WorkflowPreviewGraph } from '@features/workflow/ui/contracts';
import type { XYPosition } from '@xyflow/react';

import { Box, Portal } from '@chakra-ui/react';
import { useCallback, useRef } from 'react';

import { GraphPreviewFlow } from './GraphPreviewFlow';

/** Rendered at the workflow thumbnail's 3:2 shape; the server keeps a 256px copy, so this leaves room to downscale. */
const SNAPSHOT_WIDTH_PX = 960;
const SNAPSHOT_HEIGHT_PX = 640;
const SNAPSHOT_TIMEOUT_MS = 15_000;
/** Off-screen but laid out and painted: `html-to-image` copies computed styles, which hidden elements lose. */
const STAGE_STYLE = {
  height: `${SNAPSHOT_HEIGHT_PX}px`,
  left: '-10000px',
  pointerEvents: 'none',
  position: 'fixed',
  top: 0,
  width: `${SNAPSHOT_WIDTH_PX}px`,
} as const;

const nextFrame = () =>
  new Promise<void>((resolve) => {
    requestAnimationFrame(() => resolve());
  });

/** Captures the graph surface, not the off-screen stage around it: a clone of the fixed stage paints nothing. */
const captureSurface = async (surface: HTMLElement): Promise<Blob> => {
  // Loaded on first use: only a snapshot needs the rasterizer.
  const { toBlob } = await import('html-to-image');
  let timeout: number | undefined;
  const blob = await Promise.race([
    toBlob(surface, {
      backgroundColor: getComputedStyle(surface).backgroundColor,
      height: SNAPSHOT_HEIGHT_PX,
      pixelRatio: 1,
      width: SNAPSHOT_WIDTH_PX,
    }),
    new Promise<never>((_resolve, reject) => {
      timeout = window.setTimeout(() => reject(new Error('Timed out capturing the graph.')), SNAPSHOT_TIMEOUT_MS);
    }),
  ]).finally(() => window.clearTimeout(timeout));

  if (!blob) {
    throw new Error('The graph could not be captured.');
  }

  return blob;
};

/**
 * Renders a graph preview off-screen, fits it, and hands back a PNG of it once: the "photo" of a workflow that the
 * legacy editor's camera exported (#9501), used here for library thumbnails. Mount it to capture; unmount after.
 */
export const GraphPreviewSnapshot = ({
  graph,
  positionHints,
  onCapture,
  onError,
}: {
  graph: WorkflowPreviewGraph;
  positionHints?: Record<string, XYPosition>;
  onCapture: (image: Blob) => void;
  onError: (error: unknown) => void;
}) => {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const hasStartedRef = useRef(false);

  const handleInit = useCallback(
    (instance: { fitView: (options?: { padding?: number }) => unknown }) => {
      if (hasStartedRef.current) {
        return;
      }
      hasStartedRef.current = true;

      void (async () => {
        try {
          instance.fitView({ padding: 0.08 });
          // One frame for the viewport transform, one for the nodes to paint at it.
          await nextFrame();
          await nextFrame();

          const surface = stageRef.current?.firstElementChild;

          if (!(surface instanceof HTMLElement)) {
            throw new Error('The graph preview unmounted before it could be captured.');
          }

          onCapture(await captureSurface(surface));
        } catch (error) {
          onError(error);
        }
      })();
    },
    [onCapture, onError]
  );

  return (
    <Portal>
      <Box ref={stageRef} aria-hidden style={STAGE_STYLE}>
        <GraphPreviewFlow graph={graph} positionHints={positionHints} onInit={handleInit} />
      </Box>
    </Portal>
  );
};
