import { useStore } from '@nanostores/react';
import { getStore, useAppStore } from 'app/store/nanostores/store';
import type { AppDispatch, AppGetState, RootState } from 'app/store/store';
import { forEach, intersection, some } from 'es-toolkit/compat';
import { entityDeleted } from 'features/controlLayers/store/canvasSlice';
import {
  refImageImageChanged,
  selectReferenceImageEntities,
  selectRefImagesSlice,
} from 'features/controlLayers/store/refImagesSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { CanvasState, RefImagesState } from 'features/controlLayers/store/types';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { fieldImageCollectionValueChanged, fieldImageValueChanged } from 'features/nodes/store/nodesSlice';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { NodesState } from 'features/nodes/store/types';
import { isImageFieldCollectionInputInstance, isImageFieldInputInstance } from 'features/nodes/types/field';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { selectUpscaleSlice, type UpscaleState } from 'features/parameters/store/upscaleSlice';
import { selectSystemShouldConfirmOnDelete } from 'features/system/store/systemSlice';
import { atom } from 'nanostores';
import { useMemo } from 'react';
import { imagesApi } from 'services/api/endpoints/images';
import type { Param0 } from 'tsafe';

// Implements an awaitable modal dialog for deleting images

type DeleteImagesModalState = {
  image_names: string[];
  usagePerImage: ImageUsage[];
  usageSummary: ImageUsage;
  isOpen: boolean;
  resolve?: () => void;
  reject?: (reason?: string) => void;
};

const getInitialState = (): DeleteImagesModalState => ({
  image_names: [],
  usagePerImage: [],
  usageSummary: {
    isControlLayerImage: false,
    isInpaintMaskImage: false,
    isNodesImage: false,
    isRasterLayerImage: false,
    isRegionalGuidanceImage: false,
    isReferenceImage: false,
    isUpscaleImage: false,
  },
  isOpen: false,
});

const $deleteModalState = atom<DeleteImagesModalState>(getInitialState());

const deleteImagesWithDialog = async (image_names: string[]): Promise<void> => {
  const { getState, dispatch } = getStore();
  const imageUsage = getImageUsageFromImageNames(image_names, getState());
  const shouldConfirmOnDelete = selectSystemShouldConfirmOnDelete(getState());

  if (!shouldConfirmOnDelete && !isAnyImageInUse(imageUsage)) {
    // If we don't need to confirm and the images are not in use, delete them directly
    await handleDeletions(image_names, dispatch, getState);
  }

  return new Promise<void>((resolve, reject) => {
    $deleteModalState.set({
      usagePerImage: imageUsage,
      usageSummary: getImageUsageSummary(imageUsage),
      image_names,
      isOpen: true,
      resolve,
      reject,
    });
  });
};

const handleDeletions = async (image_names: string[], dispatch: AppDispatch, getState: AppGetState) => {
  try {
    const state = getState();
    await dispatch(imagesApi.endpoints.deleteImages.initiate({ image_names }, { track: false })).unwrap();

    if (intersection(state.gallery.selection, image_names).length > 0) {
      // Some selected images were deleted, clear selection
      dispatch(imageSelected(null));
    }

    // We need to reset the features where the image is in use - none of these work if their image(s) don't exist
    for (const image_name of image_names) {
      deleteNodesImages(state, dispatch, image_name);
      deleteControlLayerImages(state, dispatch, image_name);
      deleteReferenceImages(state, dispatch, image_name);
      deleteRasterLayerImages(state, dispatch, image_name);
    }
  } catch {
    // no-op
  }
};

const confirmDeletion = async (dispatch: AppDispatch, getState: AppGetState) => {
  const state = $deleteModalState.get();
  await handleDeletions(state.image_names, dispatch, getState);
  state.resolve?.();
  closeSilently();
};

const cancelDeletion = () => {
  const state = $deleteModalState.get();
  state.reject?.('User canceled');
  closeSilently();
};

const closeSilently = () => {
  $deleteModalState.set(getInitialState());
};

export const useDeleteImageModalState = () => {
  const state = useStore($deleteModalState);
  return state;
};

export const useDeleteImageModalApi = () => {
  const { dispatch, getState } = useAppStore();
  const api = useMemo(
    () => ({
      delete: deleteImagesWithDialog,
      confirm: () => confirmDeletion(dispatch, getState),
      cancel: cancelDeletion,
      close: closeSilently,
      getUsageSummary: getImageUsageSummary,
    }),
    [dispatch, getState]
  );

  return api;
};

const getImageUsageFromImageNames = (image_names: string[], state: RootState): ImageUsage[] => {
  if (image_names.length === 0) {
    return [];
  }

  const nodes = selectNodesSlice(state);
  const canvas = selectCanvasSlice(state);
  const upscale = selectUpscaleSlice(state);
  const refImages = selectRefImagesSlice(state);

  return image_names.map((image_name) => getImageUsage(nodes, canvas, upscale, refImages, image_name));
};

const getImageUsageSummary = (imageUsage: ImageUsage[]): ImageUsage => ({
  isUpscaleImage: some(imageUsage, (i) => i.isUpscaleImage),
  isRasterLayerImage: some(imageUsage, (i) => i.isRasterLayerImage),
  isInpaintMaskImage: some(imageUsage, (i) => i.isInpaintMaskImage),
  isRegionalGuidanceImage: some(imageUsage, (i) => i.isRegionalGuidanceImage),
  isNodesImage: some(imageUsage, (i) => i.isNodesImage),
  isControlLayerImage: some(imageUsage, (i) => i.isControlLayerImage),
  isReferenceImage: some(imageUsage, (i) => i.isReferenceImage),
});

const isAnyImageInUse = (imageUsage: ImageUsage[]): boolean =>
  imageUsage.some(
    (i) =>
      i.isRasterLayerImage ||
      i.isControlLayerImage ||
      i.isReferenceImage ||
      i.isInpaintMaskImage ||
      i.isUpscaleImage ||
      i.isNodesImage ||
      i.isRegionalGuidanceImage
  );

// Some utils to delete images from different parts of the app
const deleteNodesImages = (state: RootState, dispatch: AppDispatch, image_name: string) => {
  const actions: Param0<typeof dispatch>[] = [];
  state.nodes.present.nodes.forEach((node) => {
    if (!isInvocationNode(node)) {
      return;
    }

    forEach(node.data.inputs, (input) => {
      if (isImageFieldInputInstance(input) && input.value?.image_name === image_name) {
        actions.push(
          fieldImageValueChanged({
            nodeId: node.data.id,
            fieldName: input.name,
            value: undefined,
          })
        );
        return;
      }
      if (isImageFieldCollectionInputInstance(input)) {
        actions.push(
          fieldImageCollectionValueChanged({
            nodeId: node.data.id,
            fieldName: input.name,
            value: input.value?.filter((value) => value?.image_name !== image_name),
          })
        );
      }
    });
  });

  actions.forEach(dispatch);
};

const deleteControlLayerImages = (state: RootState, dispatch: AppDispatch, image_name: string) => {
  selectCanvasSlice(state).controlLayers.entities.forEach(({ id, objects }) => {
    let shouldDelete = false;
    for (const obj of objects) {
      if (obj.type === 'image' && 'image_name' in obj.image && obj.image.image_name === image_name) {
        shouldDelete = true;
        break;
      }
    }
    if (shouldDelete) {
      dispatch(entityDeleted({ entityIdentifier: { id, type: 'control_layer' } }));
    }
  });
};

const deleteReferenceImages = (state: RootState, dispatch: AppDispatch, image_name: string) => {
  selectReferenceImageEntities(state).forEach((entity) => {
    if (entity.config.image?.image_name === image_name) {
      dispatch(refImageImageChanged({ id: entity.id, imageDTO: null }));
    }
  });
};

const deleteRasterLayerImages = (state: RootState, dispatch: AppDispatch, image_name: string) => {
  selectCanvasSlice(state).rasterLayers.entities.forEach(({ id, objects }) => {
    let shouldDelete = false;
    for (const obj of objects) {
      if (obj.type === 'image' && 'image_name' in obj.image && obj.image.image_name === image_name) {
        shouldDelete = true;
        break;
      }
    }
    if (shouldDelete) {
      dispatch(entityDeleted({ entityIdentifier: { id, type: 'raster_layer' } }));
    }
  });
};

export const getImageUsage = (
  nodes: NodesState,
  canvas: CanvasState,
  upscale: UpscaleState,
  refImages: RefImagesState,
  image_name: string
) => {
  const isNodesImage = nodes.nodes.filter(isInvocationNode).some((node) =>
    some(node.data.inputs, (input) => {
      if (isImageFieldInputInstance(input)) {
        if (input.value?.image_name === image_name) {
          return true;
        }
      }

      if (isImageFieldCollectionInputInstance(input)) {
        if (input.value?.some((value) => value?.image_name === image_name)) {
          return true;
        }
      }

      return false;
    })
  );

  const isUpscaleImage = upscale.upscaleInitialImage?.image_name === image_name;

  const isReferenceImage = refImages.entities.some(({ config }) => config.image?.image_name === image_name);

  const isRasterLayerImage = canvas.rasterLayers.entities.some(({ objects }) =>
    objects.some((obj) => obj.type === 'image' && 'image_name' in obj.image && obj.image.image_name === image_name)
  );

  const isControlLayerImage = canvas.controlLayers.entities.some(({ objects }) =>
    objects.some((obj) => obj.type === 'image' && 'image_name' in obj.image && obj.image.image_name === image_name)
  );

  const isInpaintMaskImage = canvas.inpaintMasks.entities.some(({ objects }) =>
    objects.some((obj) => obj.type === 'image' && 'image_name' in obj.image && obj.image.image_name === image_name)
  );

  const isRegionalGuidanceImage = canvas.regionalGuidance.entities.some(({ referenceImages }) =>
    referenceImages.some(({ config }) => config.image?.image_name === image_name)
  );

  const imageUsage: ImageUsage = {
    isUpscaleImage,
    isRasterLayerImage,
    isInpaintMaskImage,
    isRegionalGuidanceImage,
    isNodesImage,
    isControlLayerImage,
    isReferenceImage,
  };

  return imageUsage;
};
