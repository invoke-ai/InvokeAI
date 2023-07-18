import { createAppAsyncThunk } from 'app/store/storeUtils';
import { selectFilteredImages } from 'features/gallery/store/gallerySelectors';
import {
  ASSETS_CATEGORIES,
  IMAGE_CATEGORIES,
} from 'features/gallery/store/gallerySlice';
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

type GetImageMetadataArg =
  paths['/api/v1/images/{image_name}']['get']['parameters']['path'];

type GetImageMetadataResponse =
  paths['/api/v1/images/{image_name}']['get']['responses']['200']['content']['application/json'];

type GetImageMetadataThunkConfig = {
  rejectValue: {
    arg: GetImageMetadataArg;
    error: unknown;
  };
};

export const imageDTOReceived = createAppAsyncThunk<
  GetImageMetadataResponse,
  GetImageMetadataArg,
  GetImageMetadataThunkConfig
>('thunkApi/imageMetadataReceived', async (arg, { rejectWithValue }) => {
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

    const images = selectFilteredImages(state);
    const categories =
      state.gallery.galleryView === 'images'
        ? IMAGE_CATEGORIES
        : ASSETS_CATEGORIES;

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
