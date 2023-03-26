import { Flex, Image, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import {
  setProgress,
  setProgressImage,
  setStatus,
  STATUS,
} from 'services/apiSlice';
import { useCallback, useEffect, useState } from 'react';
import {
  GeneratorProgressEvent,
  GraphExecutionStateCompleteEvent,
  InvocationCompleteEvent,
  InvocationStartedEvent,
} from 'services/events/types';
import {
  cancelProcessing,
  createSession,
  invokeSession,
} from 'services/thunks/session';
import { io } from 'socket.io-client';
import { useAppDispatch, useAppSelector } from './storeHooks';
import { RootState } from './store';

const socket_url = `ws://${window.location.host}`;
const socket = io(socket_url, {
  path: '/ws/socket.io',
});

const NodeAPITest = () => {
  const dispatch = useAppDispatch();
  const { sessionId, progress, progressImage } = useAppSelector(
    (state: RootState) => state.api
  );

  const [resultImages, setResultImages] = useState<string[]>([]);

  const appendResultImage = useCallback(
    (url: string) => {
      setResultImages([...resultImages, url]);
    },
    [resultImages]
  );

  const handleCreateSession = () => {
    dispatch(
      createSession({
        nodes: {
          a: {
            id: 'a',
            type: 'txt2img',
            prompt: 'pizza',
            steps: 30,
          },
          b: {
            id: 'b',
            type: 'img2img',
            prompt: 'dog',
            steps: 30,
            strength: 0.75,
          },
          c: {
            id: 'c',
            type: 'img2img',
            prompt: 'cat',
            steps: 30,
            strength: 0.75,
          },
          d: {
            id: 'd',
            type: 'img2img',
            prompt: 'jalapeno',
            steps: 30,
            strength: 0.75,
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
          {
            source: { node_id: 'c', field: 'image' },
            destination: { node_id: 'd', field: 'image' },
          },
        ],
      })
    );
  };

  const handleInvokeSession = () => {
    if (!sessionId) {
      return;
    }

    dispatch(invokeSession({ sessionId }));
    setResultImages([]);
  };

  const handleCancelProcessing = () => {
    if (!sessionId) {
      return;
    }

    dispatch(cancelProcessing({ sessionId }));
  };

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    // set up socket.io listeners

    // TODO: suppose this should be handled in the socket.io middleware?

    // subscribe to the current session
    socket.emit('subscribe', { session: sessionId });
    console.log('subscribe', { session: sessionId });

    () => {
      // cleanup
      socket.emit('unsubscribe', { session: sessionId });
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, [dispatch, sessionId]);

  useEffect(() => {
    /**
     * `invocation_started`
     */
    socket.on('invocation_started', (data: InvocationStartedEvent) => {
      console.log('invocation_started', data);
      dispatch(setStatus(STATUS.busy));
    });

    /**
     * `generator_progress`
     */
    socket.on('generator_progress', (data: GeneratorProgressEvent) => {
      console.log('generator_progress', data);
      dispatch(setProgress(data.step / data.total_steps));
      if (data.progress_image) {
        dispatch(setProgressImage(data.progress_image));
      }
    });

    /**
     * `invocation_complete`
     */
    socket.on('invocation_complete', (data: InvocationCompleteEvent) => {
      if (data.result.type === 'image') {
        const url = `api/v1/images/${data.result.image.image_type}/${data.result.image.image_name}`;
        appendResultImage(url);
      }

      console.log('invocation_complete', data);
      dispatch(setProgress(null));
      dispatch(setStatus(STATUS.idle));
      console.log(data);
    });

    /**
     * `graph_execution_state_complete`
     */
    socket.on(
      'graph_execution_state_complete',
      (data: GraphExecutionStateCompleteEvent) => {
        console.log(data);
      }
    );
  }, [dispatch, appendResultImage]);

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
      <IAIButton onClick={handleCancelProcessing} colorScheme="error">
        Cancel Processing
      </IAIButton>
      <IAIButton onClick={handleCreateSession} colorScheme="accent">
        Create Session
      </IAIButton>
      <IAIButton
        onClick={handleInvokeSession}
        loadingText={`Invoking ${
          progress === null ? '...' : `${Math.round(progress * 100)}%`
        }`}
        colorScheme="accent"
      >
        Invoke
      </IAIButton>
      <Flex wrap="wrap" gap={4} overflow="scroll">
        <Image
          src={progressImage?.dataURL}
          width={progressImage?.width}
          height={progressImage?.height}
          sx={{
            imageRendering: 'pixelated',
          }}
        />
        {resultImages.map((url) => (
          <Image key={url} src={url} />
        ))}
      </Flex>
    </Flex>
  );
};

export default NodeAPITest;
