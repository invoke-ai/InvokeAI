import { Middleware } from '@reduxjs/toolkit';
import { io } from 'socket.io-client';

import makeSocketIOEmitters from './emitters';
import makeSocketIOListeners from './listeners';

import {
  GeneratorProgressEvent,
  InvocationCompleteEvent,
  InvocationErrorEvent,
  InvocationStartedEvent,
} from 'services/events/types';
import { invocationComplete } from './actions';

const socket_url = `ws://${window.location.host}`;

const socketio = io(socket_url, {
  timeout: 60000,
  path: '/ws/socket.io',
});

export const socketioMiddleware = () => {
  let areListenersSet = false;

  const middleware: Middleware = (store) => (next) => (action) => {
    const { emitSubscribe, emitUnsubscribe } = makeSocketIOEmitters(socketio);

    const {
      onConnect,
      onDisconnect,
      onInvocationStarted,
      onGeneratorProgress,
      onInvocationError,
      onInvocationComplete,
    } = makeSocketIOListeners(store);

    if (!areListenersSet) {
      socketio.on('connect', () => onConnect());
      socketio.on('disconnect', () => onDisconnect());
    }

    areListenersSet = true;

    // use the action's match() function for type narrowing and safety
    if (invocationComplete.match(action)) {
      emitUnsubscribe(action.payload.data.graph_execution_state_id);
      socketio.removeAllListeners();
    }

    /**
     * Handle redux actions caught by middleware.
     */
    switch (action.type) {
      case 'socketio/subscribe': {
        emitSubscribe(action.payload);

        socketio.on('invocation_started', (data: InvocationStartedEvent) =>
          onInvocationStarted(data)
        );
        socketio.on('generator_progress', (data: GeneratorProgressEvent) =>
          onGeneratorProgress(data)
        );
        socketio.on('invocation_error', (data: InvocationErrorEvent) =>
          onInvocationError(data)
        );
        socketio.on('invocation_complete', (data: InvocationCompleteEvent) =>
          onInvocationComplete(data)
        );

        break;
      }

      // case 'socketio/unsubscribe': {
      //   emitUnsubscribe(action.payload);

      //   socketio.removeAllListeners();
      //   break;
      // }
    }

    next(action);
  };

  return middleware;
};
