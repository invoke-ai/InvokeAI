import { canvasPersistDenylist } from 'features/canvas/store/canvasPersistDenylist';
import { galleryPersistDenylist } from 'features/gallery/store/galleryPersistDenylist';
import { lightboxPersistDenylist } from 'features/lightbox/store/lightboxPersistDenylist';
import { nodesPersistDenylist } from 'features/nodes/store/nodesPersistDenylist';
import { generationPersistDenylist } from 'features/parameters/store/generationPersistDenylist';
import { postprocessingPersistDenylist } from 'features/parameters/store/postprocessingPersistDenylist';
import { modelsPersistDenylist } from 'features/system/store/modelsPersistDenylist';
import { systemPersistDenylist } from 'features/system/store/systemPersistDenylist';
import { uiPersistDenylist } from 'features/ui/store/uiPersistDenylist';
import { omit } from 'lodash-es';
import { SerializeFunction } from 'redux-remember';

const serializationDenylist: {
  [key: string]: string[];
} = {
  canvas: canvasPersistDenylist,
  gallery: galleryPersistDenylist,
  generation: generationPersistDenylist,
  lightbox: lightboxPersistDenylist,
  models: modelsPersistDenylist,
  nodes: nodesPersistDenylist,
  postprocessing: postprocessingPersistDenylist,
  system: systemPersistDenylist,
  // config: configPersistDenyList,
  ui: uiPersistDenylist,
  // hotkeys: hotkeysPersistDenylist,
};

export const serialize: SerializeFunction = (data, key) => {
  const result = omit(data, serializationDenylist[key]);
  return JSON.stringify(result);
};
