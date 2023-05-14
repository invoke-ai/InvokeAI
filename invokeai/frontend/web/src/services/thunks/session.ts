import { createAppAsyncThunk } from 'app/store/storeUtils';
import { SessionsService } from 'services/api';
import { log } from 'app/logging/useLogger';
import { serializeError } from 'serialize-error';

const sessionLog = log.child({ namespace: 'session' });

type SessionCreatedArg = {
  graph: Parameters<
    (typeof SessionsService)['createSession']
  >[0]['requestBody'];
};

/**
 * `SessionsService.createSession()` thunk
 */
export const sessionCreated = createAppAsyncThunk(
  'api/sessionCreated',
  async (arg: SessionCreatedArg, { rejectWithValue }) => {
    try {
      const response = await SessionsService.createSession({
        requestBody: arg.graph,
      });
      sessionLog.info({ arg, response }, `Session created (${response.id})`);
      return response;
    } catch (err: any) {
      sessionLog.error(
        {
          error: serializeError(err),
        },
        'Problem creating session'
      );
      return rejectWithValue(err.message);
    }
  }
);

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

    sessionLog.info({ arg, response }, `Node added (${response})`);

    return response;
  }
);

/**
 * `SessionsService.invokeSession()` thunk
 */
export const sessionInvoked = createAppAsyncThunk(
  'api/sessionInvoked',
  async (arg: { sessionId: string }, { rejectWithValue }) => {
    const { sessionId } = arg;

    try {
      const response = await SessionsService.invokeSession({
        sessionId,
        all: true,
      });
      sessionLog.info({ arg, response }, `Session invoked (${sessionId})`);

      return response;
    } catch (error) {
      const err = error as any;
      if (err.status === 403) {
        return rejectWithValue(err.body.detail);
      }
      throw error;
    }
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

    sessionLog.info({ arg, response }, `Session canceled (${sessionId})`);

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

    sessionLog.info(
      { arg, response },
      `Sessions listed (${response.items.length})`
    );

    return response;
  }
);
