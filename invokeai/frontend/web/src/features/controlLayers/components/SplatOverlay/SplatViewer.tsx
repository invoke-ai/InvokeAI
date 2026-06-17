import { Box } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { SplatScene } from 'features/controlLayers/components/SplatOverlay/splatScene';
import { memo, useEffect, useRef } from 'react';

const log = logger('canvas');

type Props = {
  assetUrl: string;
  /** width/height of the target raster-layer footprint, used to frame the splat (so preview ≈ commit). */
  aspect: number;
  /** Hand the live SplatScene to the parent so it can capture on commit; called with null on unmount. */
  onSceneReady: (scene: SplatScene | null) => void;
};

/**
 * The heavy three.js + Spark viewer. Loaded lazily (React.lazy) by CanvasSplatOverlay so the
 * `three` / `@sparkjsdev/spark` chunk is only fetched when the 3D overlay opens. Default export required.
 */
const SplatViewer = ({ assetUrl, aspect, onSceneReady }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const scene = new SplatScene(container);
    onSceneReady(scene);
    scene.loadFromUrl(assetUrl).catch((error: unknown) => {
      log.error({ error: String(error) }, 'Failed to load splat');
    });
    return () => {
      onSceneReady(null);
      scene.dispose();
    };
  }, [assetUrl, onSceneReady]);

  // Aspect-ratio box centered by the parent, so the live framing matches what commit captures.
  return (
    <Box ref={containerRef} position="relative" height="85%" maxWidth="90%" sx={{ aspectRatio: String(aspect) }} />
  );
};

export default memo(SplatViewer);
