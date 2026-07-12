import { Box } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { SplatScene } from 'features/controlLayers/components/SplatOverlay/splatScene';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

const log = logger('canvas');

type Props = {
  assetUrl: string;
  /** The canvas stage scale, folded into the renderer's pixel ratio so the viewport stays crisp at any zoom. */
  stageScale: number;
  /** Hand the live SplatScene to the parent so it can capture on commit; called with null on unmount. */
  onSceneReady: (scene: SplatScene | null) => void;
};

/**
 * The heavy three.js + Spark viewer, filling the splat overlay's footprint frame. Loaded lazily (React.lazy)
 * by CanvasSplatOverlay so the `three` / `@sparkjsdev/spark` chunk is only fetched when the 3D overlay opens.
 * Default export required.
 */
const SplatViewer = ({ assetUrl, stageScale, onSceneReady }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<SplatScene | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    const scene = new SplatScene(container);
    setScene(scene);
    onSceneReady(scene);
    scene.loadFromUrl(assetUrl).catch((error: unknown) => {
      log.error({ error: String(error) }, 'Failed to load splat');
    });
    return () => {
      onSceneReady(null);
      setScene(null);
      scene.dispose();
    };
  }, [assetUrl, onSceneReady]);

  useEffect(() => {
    scene?.setStageScale(stageScale);
  }, [scene, stageScale]);

  const onPointerEnter = useCallback(() => scene?.setGizmoVisible(true), [scene]);
  const onPointerLeave = useCallback(() => scene?.setGizmoVisible(false), [scene]);

  return (
    <Box
      ref={containerRef}
      position="absolute"
      inset={0}
      onPointerEnter={onPointerEnter}
      onPointerLeave={onPointerLeave}
    />
  );
};

export default memo(SplatViewer);
