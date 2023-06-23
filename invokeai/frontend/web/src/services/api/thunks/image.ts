import { createAppAsyncThunk } from 'app/store/storeUtils';
import { selectImagesAll } from 'features/gallery/store/imagesSlice';
import { size } from 'lodash-es';
import { paths } from 'services/api/schema';
import { $client } from 'services/api/client';

const { get, post, patch, del } = $client.get();

type GetImageUrlsArg =
  paths['/api/v1/images/{image_name}/urls']['get']['parameters']['path'];

type GetImageUrlsResponse =
  paths['/api/v1/images/{image_name}/urls']['get']['responses']['200']['content']['application/json'];

type GetImageUrlsThunkConfig = {
  rejectValue: {
    arg: GetImageUrlsArg;
    error: unknown;
  };
};
/**
 * Thunk to get image URLs
 */
export const imageUrlsReceived = createAppAsyncThunk<
  GetImageUrlsResponse,
  GetImageUrlsArg,
  GetImageUrlsThunkConfig
>('api/imageUrlsReceived', async (arg, { rejectWithValue }) => {
  const { image_name } = arg;
  const { data, error, response } = await get(
    '/api/v1/images/{image_name}/urls',
    {
      params: {
        path: {
          image_name,
        },
      },
    }
  );

  if (error || !data) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

type GetImageMetadataArg =
  paths['/api/v1/images/{image_name}/metadata']['get']['parameters']['path'];

type GetImageMetadataResponse =
  paths['/api/v1/images/{image_name}/metadata']['get']['responses']['200']['content']['application/json'];

type GetImageMetadataThunkConfig = {
  rejectValue: {
    arg: GetImageMetadataArg;
    error: unknown;
  };
};

export const imageMetadataReceived = createAppAsyncThunk<
  GetImageMetadataResponse,
  GetImageMetadataArg,
  GetImageMetadataThunkConfig
>('api/imageMetadataReceived', async (arg, { rejectWithValue }) => {
  const { image_name } = arg;
  const { data, error, response } = await get(
    '/api/v1/images/{image_name}/metadata',
    {
      params: {
        path: { image_name },
      },
    }
  );

  if (error || !data) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

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

type UploadImageArg =
  paths['/api/v1/images/']['post']['parameters']['query'] & {
    file: File;
    // file: paths['/api/v1/images/']['post']['requestBody']['content']['multipart/form-data']['file'];
    postUploadAction?: PostUploadAction;
  };

type UploadImageResponse =
  paths['/api/v1/images/']['post']['responses']['201']['content']['application/json'];

type UploadImageThunkConfig = {
  rejectValue: {
    arg: UploadImageArg;
    error: unknown;
  };
};
/**
 * `ImagesService.uploadImage()` thunk
 */
export const imageUploaded = createAppAsyncThunk<
  UploadImageResponse,
  UploadImageArg,
  UploadImageThunkConfig
>('api/imageUploaded', async (arg, { rejectWithValue }) => {
  const {
    postUploadAction,
    file,
    image_category,
    is_intermediate,
    session_id,
  } = arg;
  const { data, error, response } = await post('/api/v1/images/', {
    params: {
      query: {
        image_category,
        is_intermediate,
        session_id,
      },
    },
    // TODO: Proper handling of `multipart/form-data` is coming soon, will fix type issues
    // https://github.com/drwpow/openapi-typescript/issues/1123
    // @ts-ignore
    body: file,
  });

  if (error || !data) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

type DeleteImageArg =
  paths['/api/v1/images/{image_name}']['delete']['parameters']['path'];

type DeleteImageResponse =
  paths['/api/v1/images/{image_name}']['delete']['responses']['200']['content']['application/json'];

type DeleteImageThunkConfig = {
  rejectValue: {
    arg: DeleteImageArg;
    error: unknown;
  };
};
/**
 * `ImagesService.deleteImage()` thunk
 */
export const imageDeleted = createAppAsyncThunk<
  DeleteImageResponse,
  DeleteImageArg,
  DeleteImageThunkConfig
>('api/imageDeleted', async (arg, { rejectWithValue }) => {
  const { image_name } = arg;
  const { data, error, response } = await del('/api/v1/images/{image_name}', {
    params: {
      path: {
        image_name,
      },
    },
  });

  if (error || !data) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

type UpdateImageArg =
  paths['/api/v1/images/{image_name}']['patch']['requestBody']['content']['application/json'] &
    paths['/api/v1/images/{image_name}']['patch']['parameters']['path'];

type UpdateImageResponse =
  paths['/api/v1/images/{image_name}']['patch']['responses']['200']['content']['application/json'];

type UpdateImageThunkConfig = {
  rejectValue: {
    arg: UpdateImageArg;
    error: unknown;
  };
};
/**
 * `ImagesService.updateImage()` thunk
 */
export const imageUpdated = createAppAsyncThunk<
  UpdateImageResponse,
  UpdateImageArg,
  UpdateImageThunkConfig
>('api/imageUpdated', async (arg, { rejectWithValue }) => {
  const { image_name, image_category, is_intermediate, session_id } = arg;
  const { data, error, response } = await patch('/api/v1/images/{image_name}', {
    params: {
      path: {
        image_name,
      },
    },
    body: {
      image_category,
      is_intermediate,
      session_id,
    },
  });

  if (error || !data) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

export const IMAGES_PER_PAGE = 20;

const DEFAULT_IMAGES_LISTED_ARG = {
  isIntermediate: false,
  limit: IMAGES_PER_PAGE,
};

type ListImagesArg = NonNullable<
  paths['/api/v1/images/']['get']['parameters']['query']
>;

type ListImagesResponse =
  paths['/api/v1/images/']['get']['responses']['200']['content']['application/json'];

type ListImagesThunkConfig = {
  rejectValue: {
    arg: ListImagesArg;
    error: unknown;
  };
};
/**
 * `ImagesService.listImagesWithMetadata()` thunk
 */
export const receivedPageOfImages = createAppAsyncThunk<
  ListImagesResponse,
  ListImagesArg,
  ListImagesThunkConfig
>('api/receivedPageOfImages', async (arg, { getState, rejectWithValue }) => {
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

  let query: ListImagesArg = {};

  if (size(arg)) {
    query = {
      ...DEFAULT_IMAGES_LISTED_ARG,
      offset: images.length,
      ...arg,
    };
  } else {
    query = {
      ...DEFAULT_IMAGES_LISTED_ARG,
      categories,
      offset: images.length,
    };
  }

  const { data, error, response } = await get('/api/v1/images/', {
    params: {
      query,
    },
  });

  if (error || !data) {
    return rejectWithValue({ arg, error });
  }

  return data;
});
