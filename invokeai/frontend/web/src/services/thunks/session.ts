import { createAppAsyncThunk } from 'app/storeUtils';
import { SessionsService } from 'services/api';
import { buildGraph } from 'common/util/buildGraph';
import { isFulfilled } from '@reduxjs/toolkit';

type SessionCreatedArg = Parameters<
  (typeof SessionsService)['createSession']
>[0];

/**
 * `SessionsService.createSession()` thunk
 */
export const sessionCreated = createAppAsyncThunk(
  'api/sessionCreated',
  async (arg: SessionCreatedArg['requestBody'], _thunkApi) => {
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
export const isFulfilledSessionCreatedAction = isFulfilled(sessionCreated);

type NodeAddedArg = Parameters<(typeof SessionsService)['addNode']>[0];

/**
 * `SessionsService.addNode()` thunk
 */
export const nodeAdded = createAppAsyncThunk(
  'api/nodeAdded',
  async (
    arg: { node: NodeAddedArg['requestBody']; sessionId: string },
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
export const sessionInvoked = createAppAsyncThunk(
  'api/sessionInvoked',
  async (arg: { sessionId: string }, _thunkApi) => {
    const { sessionId } = arg;

    const response = await SessionsService.invokeSession({
      sessionId,
      all: true,
    });

    return response;
  }
);

type SessionCanceledArg = Parameters<
  (typeof SessionsService)['cancelSessionInvoke']
>[0];

/**
 * `SessionsService.cancelSession()` thunk
 */
export const sessionCanceled = createAppAsyncThunk(
  'api/sessionCanceled',
  async (arg: SessionCanceledArg, _thunkApi) => {
    const { sessionId } = arg;

    const response = await SessionsService.cancelSessionInvoke({
      sessionId,
    });

    return response;
  }
);

type SessionsListedArg = Parameters<
  (typeof SessionsService)['listSessions']
>[0];

/**
 * `SessionsService.listSessions()` thunk
 */
export const listedSessions = createAppAsyncThunk(
  'api/listSessions',
  async (arg: SessionsListedArg, _thunkApi) => {
    const response = await SessionsService.listSessions(arg);

    return response;
  }
);
