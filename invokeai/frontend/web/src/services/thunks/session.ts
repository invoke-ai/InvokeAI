import { createAppAsyncThunk } from 'app/storeUtils';
import { SessionsService } from 'services/api';
import { buildGraph } from 'common/util/buildGraph';
import { isFulfilled } from '@reduxjs/toolkit';

type CreateSessionArg = Parameters<
  (typeof SessionsService)['createSession']
>[0];

/**
 * `SessionsService.createSession()` thunk
 */
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
 * Function to check if an action is a fulfilled `SessionsService.createSession()` thunk
 */
export const isFulfilledCreateSession = isFulfilled(createSession);

type AddNodeArg = Parameters<(typeof SessionsService)['addNode']>[0];

/**
 * `SessionsService.addNode()` thunk
 */
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
 * `SessionsService.invokeSession()` thunk
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

type CancelSessionArg = Parameters<
  (typeof SessionsService)['cancelSessionInvoke']
>[0];

/**
 * `SessionsService.cancelSession()` thunk
 */
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

type ListSessionsArg = Parameters<(typeof SessionsService)['listSessions']>[0];

/**
 * `SessionsService.listSessions()` thunk
 */
export const listSessions = createAppAsyncThunk(
  'api/listSessions',
  async (arg: ListSessionsArg, _thunkApi) => {
    const response = await SessionsService.listSessions(arg);

    return response;
  }
);
