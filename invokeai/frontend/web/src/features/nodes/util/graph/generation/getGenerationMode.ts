import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { GenerationMode } from 'features/controlLayers/store/types';

export const getGenerationMode = async (manager?: CanvasManager | null): Promise<GenerationMode> => {
  if (!manager) {
    return 'txt2img';
  }
  return await manager.compositor.getGenerationMode();
};
