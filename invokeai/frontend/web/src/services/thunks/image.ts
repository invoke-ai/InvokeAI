import { createAppAsyncThunk } from 'app/storeUtils';
import { ImagesService, ImageType } from 'services/api';

type GetImageArg = {
  /**
   * The type of image to get
   */
  imageType: ImageType;
  /**
   * The name of the image to get
   */
  imageName: string;
};

// createAppAsyncThunk provides typing for getState and dispatch
export const getImage = createAppAsyncThunk(
  'api/getImage',
  async (arg: GetImageArg, { getState, dispatch, ...moreThunkStuff }) => {
    const response = await ImagesService.getImage(arg);
    return response;
  },
  {
    condition: (arg, { getState }) => {
      // we can get an image at any time
      return true;
    },
  }
);
