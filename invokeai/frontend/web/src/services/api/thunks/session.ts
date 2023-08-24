import { createAsyncThunk, isAnyOf } from '@reduxjs/toolkit';
import { isObject } from 'lodash-es';
import { $client } from 'services/api/client';
import { paths } from 'services/api/schema';
import { O } from 'ts-toolbelt';

type CreateSessionArg = {
  graph: NonNullable<
    paths['/api/v1/sessions/']['post']['requestBody']
  >['content']['application/json'];
};

type CreateSessionResponse = O.Required<
  NonNullable<
    paths['/api/v1/sessions/']['post']['requestBody']
  >['content']['application/json'],
  'id'
>;

type CreateSessionThunkConfig = {
  rejectValue: { arg: CreateSessionArg; status: number; error: unknown };
};

/**
 * `SessionsService.createSession()` thunk
 */
export const sessionCreated = createAsyncThunk<
  CreateSessionResponse,
  CreateSessionArg,
  CreateSessionThunkConfig
>('api/sessionCreated', async (arg, { rejectWithValue }) => {
  const { graph } = arg;
  const { POST } = $client.get();
  const { data, error, response } = await POST('/api/v1/sessions/', {
    body: graph,
  });

  if (error) {
    return rejectWithValue({ arg, status: response.status, error });
  }

  return data;
});

type InvokedSessionArg = {
  session_id: paths['/api/v1/sessions/{session_id}/invoke']['put']['parameters']['path']['session_id'];
};

type InvokedSessionResponse =
  paths['/api/v1/sessions/{session_id}/invoke']['put']['responses']['200']['content']['application/json'];

type InvokedSessionThunkConfig = {
  rejectValue: {
    arg: InvokedSessionArg;
    error: unknown;
    status: number;
  };
};

const isErrorWithStatus = (error: unknown): error is { status: number } =>
  isObject(error) && 'status' in error;

const isErrorWithDetail = (error: unknown): error is { detail: string } =>
  isObject(error) && 'detail' in error;

/**
 * `SessionsService.invokeSession()` thunk
 */
export const sessionInvoked = createAsyncThunk<
  InvokedSessionResponse,
  InvokedSessionArg,
  InvokedSessionThunkConfig
>('api/sessionInvoked', async (arg, { rejectWithValue }) => {
  const { session_id } = arg;
  const { PUT } = $client.get();
  const { error, response } = await PUT(
    '/api/v1/sessions/{session_id}/invoke',
    {
      params: { query: { all: true }, path: { session_id } },
    }
  );

  if (error) {
    if (isErrorWithStatus(error) && error.status === 403) {
      return rejectWithValue({
        arg,
        status: response.status,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        error: (error as any).body.detail,
      });
    }
    if (isErrorWithDetail(error) && response.status === 403) {
      return rejectWithValue({
        arg,
        status: response.status,
        error: error.detail,
      });
    }
    if (error) {
      return rejectWithValue({ arg, status: response.status, error });
    }
  }
});

type CancelSessionArg =
  paths['/api/v1/sessions/{session_id}/invoke']['delete']['parameters']['path'];

type CancelSessionResponse =
  paths['/api/v1/sessions/{session_id}/invoke']['delete']['responses']['200']['content']['application/json'];

type CancelSessionThunkConfig = {
  rejectValue: {
    arg: CancelSessionArg;
    error: unknown;
  };
};

/**
 * `SessionsService.cancelSession()` thunk
 */
export const sessionCanceled = createAsyncThunk<
  CancelSessionResponse,
  CancelSessionArg,
  CancelSessionThunkConfig
>('api/sessionCanceled', async (arg, { rejectWithValue }) => {
  const { session_id } = arg;
  const { DELETE } = $client.get();
  const { data, error } = await DELETE('/api/v1/sessions/{session_id}/invoke', {
    params: {
      path: { session_id },
    },
  });

  if (error) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

type ListSessionsArg = {
  params: paths['/api/v1/sessions/']['get']['parameters'];
};

type ListSessionsResponse =
  paths['/api/v1/sessions/']['get']['responses']['200']['content']['application/json'];

type ListSessionsThunkConfig = {
  rejectValue: {
    arg: ListSessionsArg;
    error: unknown;
  };
};

/**
 * `SessionsService.listSessions()` thunk
 */
export const listedSessions = createAsyncThunk<
  ListSessionsResponse,
  ListSessionsArg,
  ListSessionsThunkConfig
>('api/listSessions', async (arg, { rejectWithValue }) => {
  const { params } = arg;
  const { GET } = $client.get();
  const { data, error } = await GET('/api/v1/sessions/', {
    params,
  });

  if (error) {
    return rejectWithValue({ arg, error });
  }

  return data;
});

export const isAnySessionRejected = isAnyOf(
  sessionCreated.rejected,
  sessionInvoked.rejected
);
