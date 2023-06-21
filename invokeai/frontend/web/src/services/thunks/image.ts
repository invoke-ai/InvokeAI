import { createAppAsyncThunk } from 'app/store/storeUtils';
import { selectImagesAll } from 'features/gallery/store/imagesSlice';
import { size } from 'lodash-es';
import { ImagesService } from 'services/api';

type imageUrlsReceivedArg = Parameters<
  (typeof ImagesService)['getImageUrls']
>[0];

/**
 * `ImagesService.getImageUrls()` thunk
 */
export const imageUrlsReceived = createAppAsyncThunk(
  'api/imageUrlsReceived',
  async (arg: imageUrlsReceivedArg) => {
    const response = await ImagesService.getImageUrls(arg);
    return response;
  }
);

type imageMetadataReceivedArg = Parameters<
  (typeof ImagesService)['getImageMetadata']
>[0];

/**
 * `ImagesService.getImageUrls()` thunk
 */
export const imageMetadataReceived = createAppAsyncThunk(
  'api/imageMetadataReceived',
  async (arg: imageMetadataReceivedArg) => {
    const response = await ImagesService.getImageMetadata(arg);
    return response;
  }
);

type ControlNetAction = {
  type: 'SET_CONTROLNET_IMAGE';
  controlNetId: string;
};

type InitialImageAction = {
  type: 'SET_INITIAL_IMAGE';
};

type NodesAction = {
  type: 'SET_NODES_IMAGE';
  nodeId: string;
  fieldName: string;
};

type CanvasInitialImageAction = {
  type: 'SET_CANVAS_INITIAL_IMAGE';
};

type CanvasMergedAction = {
  type: 'TOAST_CANVAS_MERGED';
};

type CanvasSavedToGalleryAction = {
  type: 'TOAST_CANVAS_SAVED_TO_GALLERY';
};

type UploadedToastAction = {
  type: 'TOAST_UPLOADED';
};

export type PostUploadAction =
  | ControlNetAction
  | InitialImageAction
  | NodesAction
  | CanvasInitialImageAction
  | CanvasMergedAction
  | CanvasSavedToGalleryAction
  | UploadedToastAction;

type ImageUploadedArg = Parameters<(typeof ImagesService)['uploadImage']>[0] & {
  postUploadAction?: PostUploadAction;
};

/**
 * `ImagesService.uploadImage()` thunk
 */
export const imageUploaded = createAppAsyncThunk(
  'api/imageUploaded',
  async (arg: ImageUploadedArg) => {
    // `postUploadAction` is only used by the listener middleware - destructure it out
    const { postUploadAction, ...rest } = arg;
    const response = await ImagesService.uploadImage(rest);
    return response;
  }
);

type ImageDeletedArg = Parameters<(typeof ImagesService)['deleteImage']>[0];

/**
 * `ImagesService.deleteImage()` thunk
 */
export const imageDeleted = createAppAsyncThunk(
  'api/imageDeleted',
  async (arg: ImageDeletedArg) => {
    const response = await ImagesService.deleteImage(arg);
    return response;
  }
);

type ImageUpdatedArg = Parameters<(typeof ImagesService)['updateImage']>[0];

/**
 * `ImagesService.updateImage()` thunk
 */
export const imageUpdated = createAppAsyncThunk(
  'api/imageUpdated',
  async (arg: ImageUpdatedArg) => {
    const response = await ImagesService.updateImage(arg);
    return response;
  }
);

type ImagesListedArg = Parameters<
  (typeof ImagesService)['listImagesWithMetadata']
>[0];

export const IMAGES_PER_PAGE = 20;

const DEFAULT_IMAGES_LISTED_ARG = {
  isIntermediate: false,
  limit: IMAGES_PER_PAGE,
};

/**
 * `ImagesService.listImagesWithMetadata()` thunk
 */
export const receivedPageOfImages = createAppAsyncThunk(
  'api/receivedPageOfImages',
  async (arg: ImagesListedArg, { getState }) => {
    const state = getState();
    const { categories } = state.images;
    const { selectedBoardId } = state.boards;

    const images = selectImagesAll(state).filter((i) => {
      const isInCategory = categories.includes(i.image_category);
      const isInSelectedBoard = selectedBoardId
        ? i.board_id === selectedBoardId
        : true;
      return isInCategory && isInSelectedBoard;
    });

    let queryArg: ReceivedImagesArg = {};

    if (size(arg)) {
      queryArg = {
        ...DEFAULT_IMAGES_LISTED_ARG,
        offset: images.length,
        ...arg,
      };
    } else {
      queryArg = {
        ...DEFAULT_IMAGES_LISTED_ARG,
        categories,
        offset: images.length,
      };
    }

    const response = await ImagesService.listImagesWithMetadata(queryArg);
    return response;
  }
);

type ReceivedImagesArg = Parameters<
  (typeof ImagesService)['listImagesWithMetadata']
>[0];

/**
 * `ImagesService.listImagesWithMetadata()` thunk
 */
export const receivedImages = createAppAsyncThunk(
  'api/receivedImages',
  async (arg: ReceivedImagesArg, { getState }) => {
    const response = await ImagesService.listImagesWithMetadata(arg);
    return response;
  }
);
