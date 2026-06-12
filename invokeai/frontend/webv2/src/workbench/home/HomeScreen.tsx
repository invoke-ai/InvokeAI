import type { SystemStyleObject } from '@chakra-ui/react';
import { Box, Flex, Heading, HStack, Stack, Text } from '@chakra-ui/react';
import { Link, useNavigate } from '@tanstack/react-router';
import { useEffect } from 'react';
import { FileUpIcon, PlusIcon } from 'lucide-react';

import { Button } from '../components/ui/Button';
import { HomeTopBar } from './HomeTopBar';
import { ProjectsGrid } from './ProjectsGrid';
import { ResourceLinks } from './ResourceLinks';
import { toaster } from '../components/ui/toaster';
import { refreshProjectLibrary } from '../projects/library';
import { useAuthSession } from '../auth/session';
import { importProjectFile, pickProjectFile } from '../projects/projectFile';

const HOME_SCREEN_CONTENT_WRAPPER_SX: SystemStyleObject = {
  flex: 1,
  maxW: '6xl',
  mx: 'auto',
  p: { base: 4, md: 8 },
  w: 'full',
};

/**
 * The landing surface at `/`: your project library, profile, and pointers to
 * docs and community — the editor equivalent of Photoshop's home screen. It
 * deliberately mounts none of the workbench providers or runtimes, so it
 * stays on the light side of the route-level code split; everything here
 * works against the library store and the auth session alone.
 */
export const HomeScreen = () => {
  const session = useAuthSession();
  const navigate = useNavigate();

  useEffect(() => {
    void refreshProjectLibrary();
  }, []);

  const displayName = session.user?.display_name?.trim();
  const greeting = displayName ? `Welcome back, ${displayName}` : 'Welcome to Invoke';

  const handleImport = async () => {
    const file = await pickProjectFile();

    if (!file) {
      return;
    }

    try {
      const record = await importProjectFile(file);

      await navigate({ search: { project: record.project_id }, to: '/app' });
    } catch (error) {
      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: 'Import failed',
        type: 'error',
      });
    }
  };

  return (
    <Flex bg="bg" color="fg" direction="column" minH="100dvh">
      <HomeTopBar />
      <Box css={HOME_SCREEN_CONTENT_WRAPPER_SX}>
        <Flex direction={{ base: 'column', md: 'row' }} gap={{ base: '8', md: '10' }}>
          <Stack flex="1" gap="5" minW="0">
            <Flex align="center" gap="3" justify="space-between" wrap="wrap">
              <Stack gap="0.5">
                <Heading fontSize="xl" fontWeight="700">
                  {greeting}
                </Heading>
                <Text color="fg.muted" fontSize="xs">
                  Pick up where you left off, or start something new.
                </Text>
              </Stack>
              <HStack gap="2">
                <Button size="xs" variant="outline" onClick={() => void handleImport()}>
                  <FileUpIcon />
                  Import…
                </Button>
                <Button asChild size="xs" variant="solid">
                  <Link search={{ new: true }} to="/app">
                    <PlusIcon />
                    New project
                  </Link>
                </Button>
              </HStack>
            </Flex>
            <ProjectsGrid />
          </Stack>
          <Box flexShrink={0} w={{ base: 'full', md: '60' }}>
            <ResourceLinks />
          </Box>
        </Flex>
      </Box>
    </Flex>
  );
};
