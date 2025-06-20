import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { GenerationMode } from 'features/controlLayers/store/types';
import type { TabName } from 'features/ui/store/uiTypes';

export const getGenerationMode = async (manager: CanvasManager | null, tab: TabName): Promise<GenerationMode> => {
  if (!manager || tab === 'generate') {
    return 'txt2img';
  }
  return await manager.compositor.getGenerationMode();
};
