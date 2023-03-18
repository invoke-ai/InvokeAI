import { Flex, Image, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import {
  setProgress,
  setProgressImage,
  setSessionId,
  setStatus,
} from 'services/apiSlice';
import { useEffect } from 'react';
import { STATUS, ProgressImage } from 'services/apiSliceTypes';
import { getImage } from 'services/thunks/image';
import {
  cancelProcessing,
  createSession,
  invokeSession,
} from 'services/thunks/session';
import { io } from 'socket.io-client';
import { useAppDispatch, useAppSelector } from './storeHooks';
import { RootState } from './store';

type GeneratorProgress = {
  session_id: string;
  invocation_id: string;
  progress_image: ProgressImage;
  step: number;
  total_steps: number;
};

const socket_url = `ws://${window.location.host}`;
const socket = io(socket_url, {
  path: '/ws/socket.io',
});

const NodeAPITest = () => {
  const dispatch = useAppDispatch();
  const { sessionId, status, progress, progressImage } = useAppSelector(
    (state: RootState) => state.api
  );

  const handleCreateSession = () => {
    dispatch(
      createSession({
        requestBody: {
          nodes: {
            a: {
              id: 'a',
              type: 'txt2img',
              prompt: 'pizza',
              steps: 50,
              seed: 123,
            },
            b: {
              id: 'b',
              type: 'img2img',
              prompt: 'dog',
              steps: 50,
              seed: 123,
              strength: 0.9,
            },
            c: {
              id: 'c',
              type: 'img2img',
              prompt: 'cat',
              steps: 50,
              seed: 123,
              strength: 0.9,
            },
          },
          edges: [
            {
              source: { node_id: 'a', field: 'image' },
              destination: { node_id: 'b', field: 'image' },
            },
            {
              source: { node_id: 'b', field: 'image' },
              destination: { node_id: 'c', field: 'image' },
            },
          ],
        },
      })
    );
  };

  const handleInvokeSession = () => {
    dispatch(invokeSession());
  };

  const handleCancelProcessing = () => {
    dispatch(cancelProcessing());
  };

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    // set up socket.io listeners

    // TODO: suppose this should be handled in the socket.io middleware
    // TODO: write types for the socket.io payloads, haven't found a generator for them yet...

    // subscribe to the current session
    socket.emit('subscribe', { session: sessionId });
    console.log('subscribe', { session: sessionId });

    // received on each generation step
    socket.on('generator_progress', (data: GeneratorProgress) => {
      console.log('generator_progress', data);
      dispatch(setProgress(data.step / data.total_steps));
      dispatch(setProgressImage(data.progress_image));
    });

    // received after invokeSession called
    socket.on('invocation_started', (data) => {
      console.log('invocation_started', data);
      dispatch(setStatus(STATUS.busy));
    });

    // received when generation complete
    socket.on('invocation_complete', (data) => {
      // for now, just unsubscribe from the session when we finish a generation
      // in the future we will want to continue building the graph and executing etc
      console.log('invocation_complete', data);
      dispatch(setProgress(null));
      // dispatch(setSessionId(null));
      dispatch(setStatus(STATUS.idle));

      // think this gets a blob...
      // dispatch(
      //   getImage({
      //     imageType: data.result.image.image_type,
      //     imageName: data.result.image.image_name,
      //   })
      // );
    });

    // not sure when we get this?
    socket.on('session_complete', (data) => {
      // socket.emit('unsubscribe', { session: sessionId });
      // console.log('unsubscribe', { session: sessionId });
      // console.log('session_complete', data);
    });

    () => {
      // cleanup
      socket.emit('unsubscribe', { session: sessionId });
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, [dispatch, sessionId]);

  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 4,
        p: 4,
        alignItems: 'center',
        borderRadius: 'base',
      }}
    >
      <Text>Session: {sessionId ? sessionId : '...'}</Text>
      <IAIButton
        onClick={handleCancelProcessing}
        // isDisabled={!sessionId}
        colorScheme="error"
      >
        Cancel Processing
      </IAIButton>
      <IAIButton
        onClick={handleCreateSession}
        // isDisabled={status === STATUS.busy || Boolean(sessionId)}
        colorScheme="accent"
      >
        Create Session
      </IAIButton>
      <IAIButton
        onClick={handleInvokeSession}
        // isDisabled={status === STATUS.busy}
        // isLoading={status === STATUS.busy}
        loadingText={`Invoking ${
          progress === null ? '...' : `${Math.round(progress * 100)}%`
        }`}
        colorScheme="accent"
      >
        Invoke
      </IAIButton>
      <Image
        src={progressImage?.dataURL}
        width={progressImage?.width}
        height={progressImage?.height}
        sx={{
          imageRendering: 'pixelated',
        }}
      />
    </Flex>
  );
};

export default NodeAPITest;
