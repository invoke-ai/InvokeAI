import { createAppAsyncThunk } from 'app/store/storeUtils';
import { GraphExecutionState, SessionsService } from 'services/api';
import { log } from 'app/logging/useLogger';
import { isObject } from 'lodash-es';

const sessionLog = log.child({ namespace: 'session' });

type SessionCreatedArg = {
  graph: Parameters<
    (typeof SessionsService)['createSession']
  >[0]['requestBody'];
};

type SessionCreatedThunkConfig = {
  rejectValue: { arg: SessionCreatedArg; error: unknown };
};

/**
 * `SessionsService.createSession()` thunk
 */
export const sessionCreated = createAppAsyncThunk<
  GraphExecutionState,
  SessionCreatedArg,
  SessionCreatedThunkConfig
>('api/sessionCreated', async (arg, { rejectWithValue }) => {
  try {
    const response = await SessionsService.createSession({
      requestBody: arg.graph,
    });
    return response;
  } catch (error) {
    return rejectWithValue({ arg, error });
  }
});

type SessionInvokedArg = { sessionId: string };

type SessionInvokedThunkConfig = {
  rejectValue: {
    arg: SessionInvokedArg;
    error: unknown;
  };
};

const isErrorWithStatus = (error: unknown): error is { status: number } =>
  isObject(error) && 'status' in error;

/**
 * `SessionsService.invokeSession()` thunk
 */
export const sessionInvoked = createAppAsyncThunk<
  void,
  SessionInvokedArg,
  SessionInvokedThunkConfig
>('api/sessionInvoked', async (arg, { rejectWithValue }) => {
  const { sessionId } = arg;

  try {
    const response = await SessionsService.invokeSession({
      sessionId,
      all: true,
    });
    return response;
  } catch (error) {
    if (isErrorWithStatus(error) && error.status === 403) {
      return rejectWithValue({ arg, error: (error as any).body.detail });
    }
    return rejectWithValue({ arg, error });
  }
});

type SessionCanceledArg = Parameters<
  (typeof SessionsService)['cancelSessionInvoke']
>[0];
type SessionCanceledThunkConfig = {
  rejectValue: {
    arg: SessionCanceledArg;
    error: unknown;
  };
};
/**
 * `SessionsService.cancelSession()` thunk
 */
export const sessionCanceled = createAppAsyncThunk<
  void,
  SessionCanceledArg,
  SessionCanceledThunkConfig
>('api/sessionCanceled', async (arg: SessionCanceledArg, _thunkApi) => {
  const { sessionId } = arg;

  const response = await SessionsService.cancelSessionInvoke({
    sessionId,
  });

  return response;
});

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
