import { createAppAsyncThunk } from 'app/storeUtils';
import { Graph, SessionsService } from 'services/api';
import { STATUS } from 'services/apiSliceTypes';

/**
 * createSession
 */

type CreateSessionArg = { requestBody?: Graph };

// createAppAsyncThunk provides typing for getState and dispatch
export const createSession = createAppAsyncThunk(
  'api/createSession',
  async (arg: CreateSessionArg, { getState, dispatch, ...moreThunkStuff }) => {
    const response = await SessionsService.createSession(arg);
    return response;
  }
);

/**
 * invokeSession
 */

export const invokeSession = createAppAsyncThunk(
  'api/invokeSession',
  async (_arg, { getState }) => {
    const {
      api: { sessionId },
    } = getState();

    // i'd really like for the typing on the condition callback below to tell this
    // function here that sessionId will never be empty, but guess we do not get
    // that luxury
    if (!sessionId) {
      return;
    }

    const response = await SessionsService.invokeSession({
      sessionId,
      all: true,
    });

    return response;
  }
);
/**
 * invokeSession
 */

export const cancelProcessing = createAppAsyncThunk(
  'api/cancelProcessing',
  async (_arg, { getState }) => {
    console.log('before canceling');
    const {
      api: { sessionId },
    } = getState();

    // i'd really like for the typing on the condition callback below to tell this
    // function here that sessionId will never be empty, but guess we do not get
    // that luxury
    if (!sessionId) {
      return;
    }

    console.log('canceling');

    const response = await SessionsService.cancelSessionInvoke({
      sessionId,
    });

    return response;
  }
);
