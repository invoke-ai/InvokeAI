import type { SystemStyleObject } from '@chakra-ui/react';

import { Flex, Heading, HStack, Stack, Text } from '@chakra-ui/react';
import { Link, useNavigate } from '@tanstack/react-router';
import { useAuthSession } from '@workbench/auth/session';
import { Button, Scrollable, toaster } from '@workbench/components/ui';
import { ProjectsGrid } from '@workbench/launchpad/ProjectsGrid';
import { refreshProjectLibrary } from '@workbench/projects/library';
import { importProjectFile, pickProjectFile } from '@workbench/projects/projectFile';
import { FileUpIcon, PlusIcon } from 'lucide-react';
import { useEffect } from 'react';

/**
 * The Launchpad's home section: your project library. It keeps a comfortable
 * centered measure (the grid reads better than full-bleed) and owns its own
 * scroll so the rail and header stay fixed.
 */
const PROJECTS_PAGE_MEASURE_SX: SystemStyleObject = {
  maxW: '6xl',
  mx: 'auto',
  p: { base: 4, md: 8 },
  w: 'full',
};

export const ProjectsPage = () => {
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
    <Scrollable h="full" label="Projects" minH="0">
      <Stack css={PROJECTS_PAGE_MEASURE_SX} gap="5">
        <Flex align="center" gap="3" justify="space-between" wrap="wrap">
          <Stack gap="0.5">
            <Heading fontSize="xl" fontWeight="700">
              {greeting}
            </Heading>
            <Text color="fg.muted" fontSize="xs">
              Pick up where you left off, or start something new.
            </Text>
          </Stack>
          <HStack gap="2" wrap="wrap">
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
    </Scrollable>
  );
};
