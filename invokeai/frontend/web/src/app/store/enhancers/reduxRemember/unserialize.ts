import { initialCanvasState } from 'features/canvas/store/canvasSlice';
import { initialControlNetState } from 'features/controlNet/store/controlNetSlice';
import { initialGalleryState } from 'features/gallery/store/gallerySlice';
import { initialImagesState } from 'features/gallery/store/imagesSlice';
import { initialLightboxState } from 'features/lightbox/store/lightboxSlice';
import { initialNodesState } from 'features/nodes/store/nodesSlice';
import { initialGenerationState } from 'features/parameters/store/generationSlice';
import { initialPostprocessingState } from 'features/parameters/store/postprocessingSlice';
import { initialConfigState } from 'features/system/store/configSlice';
import { sd1InitialModelsState } from 'features/system/store/models/sd1ModelSlice';
import { sd2InitialModelsState } from 'features/system/store/models/sd2ModelSlice';
import { initialSystemState } from 'features/system/store/systemSlice';
import { initialHotkeysState } from 'features/ui/store/hotkeysSlice';
import { initialUIState } from 'features/ui/store/uiSlice';
import { defaultsDeep } from 'lodash-es';
import { UnserializeFunction } from 'redux-remember';

const initialStates: {
  [key: string]: any;
} = {
  canvas: initialCanvasState,
  gallery: initialGalleryState,
  generation: initialGenerationState,
  lightbox: initialLightboxState,
  sd1models: sd1InitialModelsState,
  sd2models: sd2InitialModelsState,
  nodes: initialNodesState,
  postprocessing: initialPostprocessingState,
  system: initialSystemState,
  config: initialConfigState,
  ui: initialUIState,
  hotkeys: initialHotkeysState,
  images: initialImagesState,
  controlNet: initialControlNetState,
};

export const unserialize: UnserializeFunction = (data, key) => {
  const result = defaultsDeep(JSON.parse(data), initialStates[key]);
  return result;
};
