import { createAppAsyncThunk } from 'app/store/storeUtils';
import { selectImagesAll } from 'features/gallery/store/gallerySlice';
import { size } from 'lodash-es';
import queryString from 'query-string';
import { $client } from 'services/api/client';
import { paths } from 'services/api/schema';

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
>('thunkApi/imageUrlsReceived', async (arg, { rejectWithValue }) => {
  const { image_name } = arg;
  const { get } = $client.get();
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

  if (error) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

type GetImageDTOArg =
  paths['/api/v1/images/{image_name}']['get']['parameters']['path'];

type GetImageDTOResponse =
  paths['/api/v1/images/{image_name}']['get']['responses']['200']['content']['application/json'];

type GetImageDTOThunkConfig = {
  rejectValue: {
    arg: GetImageDTOArg;
    error: unknown;
  };
};

export const imageDTOReceived = createAppAsyncThunk<
  GetImageDTOResponse,
  GetImageDTOArg,
  GetImageDTOThunkConfig
>('thunkApi/imageDTOReceived', async (arg, { rejectWithValue }) => {
  const { image_name } = arg;
  const { get } = $client.get();
  const { data, error, response } = await get('/api/v1/images/{image_name}', {
    params: {
      path: { image_name },
    },
  });

  if (error) {
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

type AddToBatchAction = {
  type: 'ADD_TO_BATCH';
};

export type PostUploadAction =
  | ControlNetAction
  | InitialImageAction
  | NodesAction
  | CanvasInitialImageAction
  | CanvasMergedAction
  | CanvasSavedToGalleryAction
  | UploadedToastAction
  | AddToBatchAction;

type UploadImageArg =
  paths['/api/v1/images/upload']['post']['parameters']['query'] & {
    file: File;
    postUploadAction?: PostUploadAction;
  };

type UploadImageResponse =
  paths['/api/v1/images/upload']['post']['responses']['201']['content']['application/json'];

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
>('thunkApi/imageUploaded', async (arg, { rejectWithValue }) => {
  const {
    postUploadAction,
    file,
    image_category,
    is_intermediate,
    session_id,
  } = arg;
  const { post } = $client.get();
  const { data, error, response } = await post('/api/v1/images/upload', {
    params: {
      query: {
        image_category,
        is_intermediate,
        session_id,
      },
    },
    body: { file },
    bodySerializer: (body) => {
      const formData = new FormData();
      formData.append('file', body.file);
      return formData;
    },
  });

  if (error) {
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
>('thunkApi/imageDeleted', async (arg, { rejectWithValue }) => {
  const { image_name } = arg;
  const { del } = $client.get();
  const { data, error, response } = await del('/api/v1/images/{image_name}', {
    params: {
      path: {
        image_name,
      },
    },
  });

  if (error) {
    return rejectWithValue({ arg, error });
  }
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
>('thunkApi/imageUpdated', async (arg, { rejectWithValue }) => {
  const { image_name, image_category, is_intermediate, session_id } = arg;
  const { patch } = $client.get();
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

  if (error) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

export const IMAGES_PER_PAGE = 20;

const DEFAULT_IMAGES_LISTED_ARG = {
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
>(
  'thunkApi/receivedPageOfImages',
  async (arg, { getState, rejectWithValue }) => {
    const { get } = $client.get();

    const state = getState();
    const { categories, selectedBoardId } = state.gallery;

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
      querySerializer: (q) => queryString.stringify(q, { arrayFormat: 'none' }),
    });

    if (error) {
      return rejectWithValue({ arg, error });
    }

    return data;
  }
);

type GetImagesByNamesArg = NonNullable<
  paths['/api/v1/images/']['post']['requestBody']['content']['application/json']
>;

type GetImagesByNamesResponse =
  paths['/api/v1/images/']['post']['responses']['200']['content']['application/json'];

type GetImagesByNamesThunkConfig = {
  rejectValue: {
    arg: GetImagesByNamesArg;
    error: unknown;
  };
};

/**
 * `ImagesService.GetImagesByNamesWithMetadata()` thunk
 */
export const receivedListOfImages = createAppAsyncThunk<
  GetImagesByNamesResponse,
  GetImagesByNamesArg,
  GetImagesByNamesThunkConfig
>(
  'thunkApi/receivedListOfImages',
  async (arg, { getState, rejectWithValue }) => {
    const { post } = $client.get();

    const { data, error, response } = await post('/api/v1/images/', {
      body: arg,
    });

    if (error) {
      return rejectWithValue({ arg, error });
    }

    return data;
  }
);
