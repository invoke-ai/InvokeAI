import { getStore } from 'app/store/nanostores/store';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import { modelsApi } from 'services/api/endpoints/models';

/**
 * Checks if the client has access to a model.
 * @param key The model key.
 * @returns A promise that resolves to true if the client has access, else false.
 */
export const checkModelAccess = async (key: string): Promise<boolean> => {
  const { dispatch } = getStore();
  try {
    const req = dispatch(modelsApi.endpoints.getModelConfig.initiate(key));
    req.unsubscribe();
    const result = await req.unwrap();
    return Boolean(result);
  } catch {
    return false;
  }
};

/**
 * Checks if the client has access to an image.
 * @param name The image name.
 * @returns A promise that resolves to true if the client has access, else false.
 */
export const checkImageAccess = async (name: string): Promise<boolean> => {
  const { dispatch } = getStore();
  try {
    const req = dispatch(imagesApi.endpoints.getImageDTO.initiate(name));
    req.unsubscribe();
    const result = await req.unwrap();
    return Boolean(result);
  } catch {
    return false;
  }
};

/**
 * Checks if the client has access to a board.
 * @param id The board id.
 * @returns A promise that resolves to true if the client has access, else false.
 */
export const checkBoardAccess = async (id: string): Promise<boolean> => {
  const { dispatch } = getStore();
  try {
    const req = dispatch(boardsApi.endpoints.listAllBoards.initiate({ include_archived: true }));
    req.unsubscribe();
    const result = await req.unwrap();
    return result.some((b) => b.board_id === id);
  } catch {
    return false;
  }
};
