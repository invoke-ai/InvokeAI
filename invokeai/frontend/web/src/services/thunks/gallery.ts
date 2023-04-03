import { createAppAsyncThunk } from 'app/storeUtils';
import { SessionsService } from 'services/api';

/**
 * Get the last 10 sessions' worth of images.
 *
 * This should be at most 10 images so long as we continue to make a new session for every
 * generation.
 *
 * If a session was created but no image generated, this will be < 10 images.
 *
 * When we allow more images per sesssion, this is kinda no longer a viable way to grab results,
 * because a session could have many, many images. In that situation, barring a change to the api,
 * we have to keep track of images we've grabbed and the session they came from, so that when we
 * want to load more, we can "resume" fetching images from that session.
 *
 * The API should change.
 */
export const getNextResultsPage = createAppAsyncThunk(
  'results/getMoreResultsImages',
  async (_arg, { getState }) => {
    const { page } = getState().results;

    const response = await SessionsService.listSessions({
      page: page + 1,
      perPage: 10,
    });

    return response;
  }
);

export const getInitialResultsPage = createAppAsyncThunk(
  'results/getMoreResultsImages',
  async (_arg) => {
    const response = await SessionsService.listSessions({
      page: 0,
      perPage: 10,
    });

    return response;
  }
);
