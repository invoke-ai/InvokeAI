import { MiddlewareAPI } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { AppDispatch, RootState } from 'app/store/store';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { Socket } from 'socket.io-client';
import {
  socketConnected,
  socketDisconnected,
  socketGeneratorProgress,
  socketGraphExecutionStateComplete,
  socketInvocationComplete,
  socketInvocationError,
  socketInvocationRetrievalError,
  socketInvocationStarted,
  socketModelLoadCompleted,
  socketModelLoadStarted,
  socketSessionRetrievalError,
  socketSubscribed,
} from '../actions';
import { ClientToServerEvents, ServerToClientEvents } from '../types';

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  storeApi: MiddlewareAPI<AppDispatch, RootState>;
};

export const setEventListeners = (arg: SetEventListenersArg) => {
  const { socket, storeApi } = arg;
  const { dispatch, getState } = storeApi;

  /**
   * Connect
   */
  socket.on('connect', () => {
    const log = logger('socketio');
    log.debug('Connected');

    dispatch(socketConnected());

    const { sessionId } = getState().system;

    if (sessionId) {
      socket.emit('subscribe', { session: sessionId });
      dispatch(
        socketSubscribed({
          sessionId,
        })
      );
    }
  });

  socket.on('connect_error', (error) => {
    if (error && error.message) {
      const data: string | undefined = (
        error as unknown as { data: string | undefined }
      ).data;
      if (data === 'ERR_UNAUTHENTICATED') {
        dispatch(
          addToast(
            makeToast({
              title: error.message,
              status: 'error',
              duration: 10000,
            })
          )
        );
      }
    }
  });

  /**
   * Disconnect
   */
  socket.on('disconnect', () => {
    dispatch(socketDisconnected());
  });

  /**
   * Invocation started
   */
  socket.on('invocation_started', (data) => {
    dispatch(socketInvocationStarted({ data }));
  });

  /**
   * Generator progress
   */
  socket.on('generator_progress', (data) => {
    dispatch(socketGeneratorProgress({ data }));
  });

  /**
   * Invocation error
   */
  socket.on('invocation_error', (data) => {
    dispatch(socketInvocationError({ data }));
  });

  /**
   * Invocation complete
   */
  socket.on('invocation_complete', (data) => {
    dispatch(
      socketInvocationComplete({
        data,
      })
    );
  });

  /**
   * Graph complete
   */
  socket.on('graph_execution_state_complete', (data) => {
    dispatch(
      socketGraphExecutionStateComplete({
        data,
      })
    );
  });

  /**
   * Model load started
   */
  socket.on('model_load_started', (data) => {
    dispatch(
      socketModelLoadStarted({
        data,
      })
    );
  });

  /**
   * Model load completed
   */
  socket.on('model_load_completed', (data) => {
    dispatch(
      socketModelLoadCompleted({
        data,
      })
    );
  });

  /**
   * Session retrieval error
   */
  socket.on('session_retrieval_error', (data) => {
    dispatch(
      socketSessionRetrievalError({
        data,
      })
    );
  });

  /**
   * Invocation retrieval error
   */
  socket.on('invocation_retrieval_error', (data) => {
    dispatch(
      socketInvocationRetrievalError({
        data,
      })
    );
  });
};
