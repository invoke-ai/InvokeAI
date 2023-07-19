import {
  PayloadAction,
  createAction,
  createSelector,
  createSlice,
} from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { some } from 'lodash-es';
import { ImageDTO } from 'services/api/types';

type DeleteImageState = {
  imageToDelete: ImageDTO | null;
  isModalOpen: boolean;
};

export const initialDeleteImageState: DeleteImageState = {
  imageToDelete: null,
  isModalOpen: false,
};

const imageDeletion = createSlice({
  name: 'imageDeletion',
  initialState: initialDeleteImageState,
  reducers: {
    isModalOpenChanged: (state, action: PayloadAction<boolean>) => {
      state.isModalOpen = action.payload;
    },
    imageToDeleteSelected: (state, action: PayloadAction<ImageDTO>) => {
      state.imageToDelete = action.payload;
    },
    imageToDeleteCleared: (state) => {
      state.imageToDelete = null;
      state.isModalOpen = false;
    },
  },
});

export const {
  isModalOpenChanged,
  imageToDeleteSelected,
  imageToDeleteCleared,
} = imageDeletion.actions;

export default imageDeletion.reducer;

export type ImageUsage = {
  isInitialImage: boolean;
  isCanvasImage: boolean;
  isNodesImage: boolean;
  isControlNetImage: boolean;
};

export const getImageUsage = (state: RootState, image_name: string) => {
  const { generation, canvas, nodes, controlNet } = state;
  const isInitialImage = generation.initialImage?.imageName === image_name;

  const isCanvasImage = canvas.layerState.objects.some(
    (obj) => obj.kind === 'image' && obj.imageName === image_name
  );

  const isNodesImage = nodes.nodes.some((node) => {
    return some(
      node.data.inputs,
      (input) =>
        input.type === 'image' && input.value?.image_name === image_name
    );
  });

  const isControlNetImage = some(
    controlNet.controlNets,
    (c) =>
      c.controlImage === image_name || c.processedControlImage === image_name
  );

  const imageUsage: ImageUsage = {
    isInitialImage,
    isCanvasImage,
    isNodesImage,
    isControlNetImage,
  };

  return imageUsage;
};

export const selectImageUsage = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { imageToDelete } = state.imageDeletion;

    if (!imageToDelete) {
      return;
    }

    const { image_name } = imageToDelete;

    const imageUsage = getImageUsage(state, image_name);

    return imageUsage;
  },
  defaultSelectorOptions
);

export const imageDeletionConfirmed = createAction<{
  imageDTO: ImageDTO;
  imageUsage: ImageUsage;
}>('imageDeletion/imageDeletionConfirmed');
