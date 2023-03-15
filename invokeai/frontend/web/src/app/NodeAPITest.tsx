import { Flex, Heading, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import { useEffect, useState } from 'react';
import { SessionsService } from 'services/api';
import { io } from 'socket.io-client';

type GeneratorProgress = {
  session_id: string;
  invocation_id: string;
  step: number;
  percent: number;
};

const socket_url = `ws://${window.location.host}`;
const socket = io(socket_url, {
  path: '/ws/socket.io',
});

enum STATUS {
  waiting = 'WAITING',
  ready = 'READY',
  preparing = 'PREPARING',
  generating = 'GENERATING',
  finished = 'FINISHED',
}

const NodeAPITest = () => {
  const [invocationProgress, setInvocationProgress] = useState<number>();
  const [status, setStatus] = useState<STATUS>(STATUS.waiting);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const handleCreateSession = async () => {
    // create a session with a simple graph
    const payload = await SessionsService.createSession({
      nodes: {
        a: {
          id: 'a',
          type: 'txt2img',
          prompt: 'pizza',
          steps: 10,
        },
        b: {
          id: 'b',
          type: 'show_image',
        },
      },
      edges: [
        {
          source: { node_id: 'a', field: 'image' },
          destination: { node_id: 'b', field: 'image' },
        },
      ],
    });

    // the generated types have `id` as optional but i'm pretty sure we always get the id
    setSessionId(payload.id!);
    setStatus(STATUS.ready);
    console.log('payload', payload);

    // subscribe to this session
    socket.emit('subscribe', { session: payload.id });
    console.log('subscribe', { session: payload.id });
  };

  const handleInvokeSession = async () => {
    if (!sessionId) {
      return;
    }

    setStatus(STATUS.preparing);
    // invoke the session, the resultant image should open in your platform's native image viewer when completed
    await SessionsService.invokeSession(sessionId, true);
  };

  useEffect(() => {
    socket.on('generator_progress', (data: GeneratorProgress) => {
      // this is broken on the backend, the nodes web server does not get `step` or `steps`, so we don't get a percentage
      // see https://github.com/invoke-ai/InvokeAI/issues/2951
      console.log('generator_progress', data);
      setInvocationProgress(data.percent);
    });
    socket.on('invocation_started', (data) => {
      console.log('invocation_started', data);
      setStatus(STATUS.generating);
    });
    socket.on('invocation_complete', (data) => {
      // for now, just unsubscribe from the session when we finish a generation
      // in the future we will want to continue building the graph and executing etc
      setStatus(STATUS.finished);
      console.log('invocation_complete', data);
      socket.emit('unsubscribe', { session: data.session_id });
      console.log('unsubscribe', { session: data.session_id });
      setTimeout(() => {
        setSessionId(null);
        setStatus(STATUS.waiting);
      }, 2000);
    });
    socket.on('session_complete', (data) => {
      console.log('session_complete', data);
      socket.emit('unsubscribe', { session: data.session_id });
      console.log('unsubscribe', { session: data.session_id });
      setSessionId(null);
      setStatus(STATUS.waiting);
    });

    () => {
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, []);

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
      <Heading size="lg">Status: {status}</Heading>
      <Text>Session: {sessionId ? sessionId : '...'}</Text>
      <IAIButton
        onClick={handleCreateSession}
        isDisabled={!!sessionId}
        colorScheme="accent"
      >
        Create Session
      </IAIButton>
      <IAIButton
        onClick={handleInvokeSession}
        isDisabled={!sessionId || status !== STATUS.ready}
        isLoading={[STATUS.preparing, STATUS.generating].includes(status)}
        loadingText={`Invoking ${
          invocationProgress === undefined
            ? '...'
            : `${Math.round(invocationProgress * 100)}%`
        }`}
        colorScheme="accent"
      >
        Invoke
      </IAIButton>
    </Flex>
  );
};

export default NodeAPITest;
