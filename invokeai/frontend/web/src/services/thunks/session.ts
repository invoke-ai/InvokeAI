import { createAppAsyncThunk } from 'app/storeUtils';
import { SessionsService } from 'services/api';
import { buildGraph } from 'common/util/buildGraph';

/**
 * createSession thunk
 */

/**
 * Extract the type of the requestBody from the generated API client.
 *
 * Would really like for this to be generated but it's easy enough to extract it.
 */

type CreateSessionArg = Parameters<
  (typeof SessionsService)['createSession']
>[0];

export const createSession = createAppAsyncThunk(
  'api/createSession',
  async (arg: CreateSessionArg['requestBody'], _thunkApi) => {
    let graph = arg;
    if (!arg) {
      const { getState } = _thunkApi;
      const state = getState();
      graph = buildGraph(state);
    }

    const response = await SessionsService.createSession({
      requestBody: graph,
    });

    return response;
  }
);

/**
 * addNode thunk
 */

type AddNodeArg = Parameters<(typeof SessionsService)['addNode']>[0];

export const addNode = createAppAsyncThunk(
  'api/addNode',
  async (
    arg: { node: AddNodeArg['requestBody']; sessionId: string },
    _thunkApi
  ) => {
    const response = await SessionsService.addNode({
      requestBody: arg.node,
      sessionId: arg.sessionId,
    });

    return response;
  }
);

/**
 * invokeSession thunk
 */

export const invokeSession = createAppAsyncThunk(
  'api/invokeSession',
  async (arg: { sessionId: string }, _thunkApi) => {
    const { sessionId } = arg;

    const response = await SessionsService.invokeSession({
      sessionId,
      all: true,
    });

    return response;
  }
);

/**
 * cancelSession thunk
 */

type CancelSessionArg = Parameters<
  (typeof SessionsService)['cancelSessionInvoke']
>[0];

export const cancelProcessing = createAppAsyncThunk(
  'api/cancelProcessing',
  async (arg: CancelSessionArg, _thunkApi) => {
    const { sessionId } = arg;

    const response = await SessionsService.cancelSessionInvoke({
      sessionId,
    });

    return response;
  }
);
