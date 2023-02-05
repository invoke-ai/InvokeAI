import ImageUploader from 'common/components/ImageUploader';
import Console from 'features/system/components/Console';
import ProgressBar from 'features/system/components/ProgressBar';
import SiteHeader from 'features/system/components/SiteHeader';
import InvokeTabs from 'features/ui/components/InvokeTabs';
import { keepGUIAlive } from './utils';

import { Button, Flex, Text } from '@chakra-ui/react';
import { SessionsService } from 'services/openapi';

import useToastWatcher from 'features/system/hooks/useToastWatcher';

import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

type GeneratorProgress = {
  session_id: string;
  invocation_id: string;
  step: number;
  percent: number;
};

keepGUIAlive();

const socket_url = `ws://${window.location.host}`;
const socket = io(socket_url, {
  path: '/ws/socket.io',
});

const App = () => {
  useToastWatcher();

  const [sessionId, setSessionId] = useState<string>();
  const [invocationProgress, setInvocationProgress] = useState<number>();
  const [isInvocationPending, setIsInvocationPending] = useState(false);

  const handleCreateSession = async () => {
    const { id } = await SessionsService.createSession({
      nodes: [
        {
          id: 'a',
          type: 'txt2img',
          prompt: 'pizza',
          steps: 10,
        },
        {
          id: 'b',
          type: 'show_image',
        },
      ],
      links: [
        {
          from_node: { id: 'a', field: 'image' },
          to_node: { id: 'b', field: 'image' },
        },
      ],
    });

    setSessionId(id);
    socket.emit('subscribe', { session: id });
  };

  const handleInvokeSession = async () => {
    if (sessionId) {
      setIsInvocationPending(true);
      SessionsService.invokeSession(sessionId, true);
    }
  };

  useEffect(() => {
    socket.removeAllListeners();
    socket.on('generator_progress', (data: GeneratorProgress) => {
      console.log('generator_progress', data);
      setInvocationProgress(data.percent);
    });
    socket.on('invocation_complete', async (data) => {
      setIsInvocationPending(false);
      setInvocationProgress(undefined);
      setSessionId(undefined);
      console.log('invocation_complete', data);
      socket.emit('unsubscribe', { session: sessionId });
    });
    socket.on('invocation_started', (data) =>
      console.log('invocation_started', data)
    );
    socket.on('session_complete', (data) => {
      console.log('session_complete', data);
    });

    () => {
      socket.emit('unsubscribe', { session: sessionId });
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, [sessionId]);

  return (
    <div className="App">
      <ImageUploader>
        <ProgressBar />
        <Flex gap={2} p={2} alignItems="center">
          <Button onClick={handleCreateSession} isDisabled={!!sessionId}>
            Create Session
          </Button>
          <Button
            onClick={handleInvokeSession}
            isDisabled={!sessionId || isInvocationPending}
            isLoading={isInvocationPending}
            loadingText={`Invoking ${
              invocationProgress === undefined
                ? '...'
                : `${Math.round(invocationProgress * 100)}%`
            }`}
          >
            Invoke
          </Button>
          {sessionId && <Text>Session: {sessionId}</Text>}
        </Flex>
        <div className="app-content">
          <SiteHeader />
          <InvokeTabs />
        </div>
        <div className="app-console">
          <Console />
        </div>
      </ImageUploader>
      <FloatingParametersPanelButtons />
      <FloatingGalleryButton />
    </div>
  );
};

export default App;
