import { createAppAsyncThunk } from 'app/storeUtils';
import { ModelsService } from 'services/api';

export const IMAGES_PER_PAGE = 20;

export const receivedModels = createAppAsyncThunk(
  'models/receivedModels',
  async (_arg) => {
    const response = await ModelsService.listModels();

    return response;
  }
);
