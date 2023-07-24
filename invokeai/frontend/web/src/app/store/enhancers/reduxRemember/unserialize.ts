import { initialCanvasState } from 'features/canvas/store/canvasSlice';
import { initialControlNetState } from 'features/controlNet/store/controlNetSlice';
import { initialGalleryState } from 'features/gallery/store/gallerySlice';
import { initialNodesState } from 'features/nodes/store/nodesSlice';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import { initialPostprocessingState } from 'features/parameters/store/postprocessingSlice';
import { initialConfigState } from 'features/system/store/configSlice';
import { initialSystemState } from 'features/system/store/systemSlice';
import { initialHotkeysState } from 'features/ui/store/hotkeysSlice';
import { initialUIState } from 'features/ui/store/uiSlice';
import { defaultsDeep } from 'lodash-es';
import { UnserializeFunction } from 'redux-remember';

const initialStates: {
  [key: string]: object; // TODO: type this properly
} = {
  canvas: initialCanvasState,
  gallery: initialGalleryState,
  generation: initialGenerationState,
  nodes: initialNodesState,
  postprocessing: initialPostprocessingState,
  system: initialSystemState,
  config: initialConfigState,
  ui: initialUIState,
  hotkeys: initialHotkeysState,
  controlNet: initialControlNetState,
};

export const unserialize: UnserializeFunction = (data, key) => {
  const result = defaultsDeep(JSON.parse(data), initialStates[key]);
  return result;
};
