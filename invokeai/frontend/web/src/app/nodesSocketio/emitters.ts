import { Socket } from 'socket.io-client';

const makeSocketIOEmitters = (socketio: Socket) => {
  return {
    emitSubscribe: (sessionId: string) => {
      socketio.emit('subscribe', { session: sessionId });
    },

    emitUnsubscribe: (sessionId: string) => {
      socketio.emit('unsubscribe', { session: sessionId });
    },
  };
};

export default makeSocketIOEmitters;
