import { MiddlewareAPI } from '@reduxjs/toolkit';
import { AppDispatch, RootState } from 'app/store/store';
import { getTimestamp } from 'common/util/getTimestamp';
import { Socket } from 'socket.io-client';
import {
  socketGeneratorProgress,
  socketGraphExecutionStateComplete,
  socketInvocationComplete,
  socketInvocationError,
  socketInvocationStarted,
  socketConnected,
  socketDisconnected,
  socketSubscribed,
} from '../actions';
import { ClientToServerEvents, ServerToClientEvents } from '../types';
import { Logger } from 'roarr';
import { JsonObject } from 'roarr/dist/types';
import { makeToast } from '../../../app/components/Toaster';
import { addToast } from '../../../features/system/store/systemSlice';

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  storeApi: MiddlewareAPI<AppDispatch, RootState>;
  log: Logger<JsonObject>;
};

export const setEventListeners = (arg: SetEventListenersArg) => {
  const { socket, storeApi, log } = arg;
  const { dispatch, getState } = storeApi;

  /**
   * Connect
   */
  socket.on('connect', () => {
    log.debug('Connected');

    dispatch(socketConnected({ timestamp: getTimestamp() }));

    const { sessionId } = getState().system;

    if (sessionId) {
      socket.emit('subscribe', { session: sessionId });
      dispatch(
        socketSubscribed({
          sessionId,
          timestamp: getTimestamp(),
        })
      );
    }
  });

  socket.on('connect_error', (error) => {
    if (error && error.message) {
      const data: string | undefined = (error as any).data;
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
    dispatch(socketDisconnected({ timestamp: getTimestamp() }));
  });

  /**
   * Invocation started
   */
  socket.on('invocation_started', (data) => {
    dispatch(socketInvocationStarted({ data, timestamp: getTimestamp() }));
  });

  /**
   * Generator progress
   */
  socket.on('generator_progress', (data) => {
    dispatch(socketGeneratorProgress({ data, timestamp: getTimestamp() }));
  });

  /**
   * Invocation error
   */
  socket.on('invocation_error', (data) => {
    dispatch(socketInvocationError({ data, timestamp: getTimestamp() }));
  });

  /**
   * Invocation complete
   */
  socket.on('invocation_complete', (data) => {
    dispatch(
      socketInvocationComplete({
        data,
        timestamp: getTimestamp(),
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
        timestamp: getTimestamp(),
      })
    );
  });
};
