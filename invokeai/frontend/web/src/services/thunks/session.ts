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
  },
  {
    // if this returns false, the api call is skipped
    // we can guard in many places, and maybe this isn't right for us,
    // but just trying it here
    condition: (arg, { getState }) => {
      const {
        api: { status, sessionId },
      } = getState();

      // don't create session if we are processing already
      if (status === STATUS.busy) {
        return false;
      }

      // don't create session if we have a sessionId
      if (sessionId) {
        return false;
      }
    },
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
  },
  {
    condition: (arg, { getState }) => {
      const {
        api: { status, sessionId },
      } = getState();

      // don't invoke if we are processing already
      if (status === STATUS.busy) {
        return false;
      }

      // don't invoke if we don't have a sessionId
      if (!sessionId) {
        return false;
      }
    },
  }
);
