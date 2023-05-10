import { MiddlewareAPI } from '@reduxjs/toolkit';
import { AppDispatch, RootState } from 'app/store/store';
import { getTimestamp } from 'common/util/getTimestamp';
import { sessionCanceled } from 'services/thunks/session';
import { Socket } from 'socket.io-client';
import {
  generatorProgress,
  graphExecutionStateComplete,
  invocationComplete,
  invocationError,
  invocationStarted,
  socketConnected,
  socketDisconnected,
  socketSubscribed,
} from '../actions';
import { ClientToServerEvents, ServerToClientEvents } from '../types';
import { Logger } from 'roarr';
import { JsonObject } from 'roarr/dist/types';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from 'services/thunks/gallery';
import { receivedModels } from 'services/thunks/model';
import { receivedOpenAPISchema } from 'services/thunks/schema';
import { makeToast } from '../../../features/system/hooks/useToastWatcher';
import { addToast } from '../../../features/system/store/systemSlice';

type SetEventListenersArg = {
  socket: Socket<ServerToClientEvents, ClientToServerEvents>;
  store: MiddlewareAPI<AppDispatch, RootState>;
  log: Logger<JsonObject>;
};

export const setEventListeners = (arg: SetEventListenersArg) => {
  const { socket, store, log } = arg;
  const { dispatch, getState } = store;

  /**
   * Connect
   */
  socket.on('connect', () => {
    log.debug('Connected');

    dispatch(socketConnected({ timestamp: getTimestamp() }));

    const { results, uploads, models, nodes, config, system } = getState();

    const { disabledTabs } = config;

    // These thunks need to be dispatch in middleware; cannot handle in a reducer
    if (!results.ids.length) {
      dispatch(receivedResultImagesPage());
    }

    if (!uploads.ids.length) {
      dispatch(receivedUploadImagesPage());
    }

    if (!models.ids.length) {
      dispatch(receivedModels());
    }

    if (!nodes.schema && !disabledTabs.includes('nodes')) {
      dispatch(receivedOpenAPISchema());
    }

    if (system.sessionId) {
      log.debug(
        { sessionId: system.sessionId },
        `Subscribed to existing session (${system.sessionId})`
      );

      socket.emit('subscribe', { session: system.sessionId });
      dispatch(
        socketSubscribed({
          sessionId: system.sessionId,
          timestamp: getTimestamp(),
        })
      );
    }
  });

  socket.on('connect_error', (error) => {
    if (error && error.message) {
      dispatch(
        addToast(
          makeToast({ title: error.message, status: 'error', duration: 10000 })
        )
      );
    }
  });

  /**
   * Disconnect
   */
  socket.on('disconnect', () => {
    log.debug('Disconnected');
    dispatch(socketDisconnected({ timestamp: getTimestamp() }));
  });

  /**
   * Invocation started
   */
  socket.on('invocation_started', (data) => {
    if (getState().system.canceledSession === data.graph_execution_state_id) {
      log.trace(
        { data, sessionId: data.graph_execution_state_id },
        `Ignored invocation started (${data.node.type}) for canceled session (${data.graph_execution_state_id})`
      );
      return;
    }

    log.info(
      { data, sessionId: data.graph_execution_state_id },
      `Invocation started (${data.node.type})`
    );
    dispatch(invocationStarted({ data, timestamp: getTimestamp() }));
  });

  /**
   * Generator progress
   */
  socket.on('generator_progress', (data) => {
    if (getState().system.canceledSession === data.graph_execution_state_id) {
      log.trace(
        { data, sessionId: data.graph_execution_state_id },
        `Ignored generator progress (${data.node.type}) for canceled session (${data.graph_execution_state_id})`
      );
      return;
    }

    log.trace(
      { data, sessionId: data.graph_execution_state_id },
      `Generator progress (${data.node.type})`
    );
    dispatch(generatorProgress({ data, timestamp: getTimestamp() }));
  });

  /**
   * Invocation error
   */
  socket.on('invocation_error', (data) => {
    log.error(
      { data, sessionId: data.graph_execution_state_id },
      `Invocation error (${data.node.type})`
    );
    dispatch(invocationError({ data, timestamp: getTimestamp() }));
  });

  /**
   * Invocation complete
   */
  socket.on('invocation_complete', (data) => {
    log.info(
      { data, sessionId: data.graph_execution_state_id },
      `Invocation complete (${data.node.type})`
    );
    const sessionId = data.graph_execution_state_id;

    const { cancelType, isCancelScheduled } = getState().system;
    const { shouldFetchImages } = getState().config;

    // Handle scheduled cancelation
    if (cancelType === 'scheduled' && isCancelScheduled) {
      dispatch(sessionCanceled({ sessionId }));
    }

    dispatch(
      invocationComplete({
        data,
        timestamp: getTimestamp(),
        shouldFetchImages,
      })
    );
  });

  /**
   * Graph complete
   */
  socket.on('graph_execution_state_complete', (data) => {
    log.info(
      { data, sessionId: data.graph_execution_state_id },
      `Graph execution state complete (${data.graph_execution_state_id})`
    );
    dispatch(graphExecutionStateComplete({ data, timestamp: getTimestamp() }));
  });
};
