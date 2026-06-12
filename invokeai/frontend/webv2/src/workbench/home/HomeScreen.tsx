import type { SystemStyleObject } from '@chakra-ui/react';
import { Box, Flex, Heading, HStack, Icon, Stack, Tabs, Text } from '@chakra-ui/react';
import { Link, useLocation, useNavigate } from '@tanstack/react-router';
import { useEffect, type ReactNode } from 'react';
import { FileUpIcon, FolderIcon, PlusIcon, UsersIcon, type LucideIcon } from 'lucide-react';

import { Button } from '../components/ui/Button';
import { HomeTopBar } from './HomeTopBar';
import { ProjectsGrid } from './ProjectsGrid';
import { ResourceLinks } from './ResourceLinks';
import { toaster } from '../components/ui/toaster';
import { refreshProjectLibrary } from '../projects/library';
import { useAuthSession } from '../auth/session';
import { importProjectFile, pickProjectFile } from '../projects/projectFile';
import { UsersManagementPanel } from '../users';

const HOME_SCREEN_CONTENT_WRAPPER_SX: SystemStyleObject = {
  flex: 1,
  maxW: '6xl',
  mx: 'auto',
  p: { base: 4, md: 8 },
  w: 'full',
};

type HomeTabId = 'projects' | 'users';

interface HomeTabDefinition {
  id: HomeTabId;
  label: string;
  icon: LucideIcon;
  children: ReactNode;
  condition?: boolean;
}

const DEFAULT_HOME_TAB_ID: HomeTabId = 'projects';

const isHomeTabId = (value: string): value is HomeTabId => value === 'projects' || value === 'users';

const normalizeHomeTabId = (value: string): HomeTabId | null => {
  const id = value.replace(/^\/+/, '');

  return isHomeTabId(id) ? id : null;
};

const getRequestedHomeTabId = (pathname: string, hash: string): HomeTabId | null =>
  normalizeHomeTabId(pathname) ?? normalizeHomeTabId(hash);

const getActiveHomeTabId = (tabs: HomeTabDefinition[], requestedTabId: HomeTabId | null): HomeTabId =>
  requestedTabId && tabs.some((tab) => tab.id === requestedTabId) ? requestedTabId : DEFAULT_HOME_TAB_ID;

const replaceHomeTabHash = (tabId: HomeTabId): void => {
  window.history.replaceState(
    window.history.state,
    '',
    `${window.location.pathname}${window.location.search}#${tabId}`
  );
};

const pushHomeTabHash = (tabId: HomeTabId): void => {
  if (window.location.hash !== `#${tabId}`) {
    window.location.hash = tabId;
  }
};

/**
 * The landing surface at `/`: your project library, profile/admin controls,
 * and pointers to docs and community — the editor equivalent of Photoshop's
 * home screen. It deliberately mounts none of the workbench providers or
 * runtimes, so it stays on the light side of the route-level code split.
 */
export const HomeScreen = () => {
  const session = useAuthSession();
  const navigate = useNavigate();

  useEffect(() => {
    void refreshProjectLibrary();
  }, []);

  const displayName = session.user?.display_name?.trim();
  const greeting = displayName ? `Welcome back, ${displayName}` : 'Welcome to Invoke';
  const canManageUsers = session.multiuserEnabled && session.user?.is_admin === true;

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

  const allTabs: HomeTabDefinition[] = [
    {
      label: 'Projects',
      icon: FolderIcon,
      id: 'projects',
      children: <ProjectsPanel greeting={greeting} onImport={handleImport} />,
    },
    {
      label: 'Manage Users',
      icon: UsersIcon,
      id: 'users',
      children: <UsersPanel />,
      condition: canManageUsers,
    },
  ];
  const tabs = allTabs.filter((tab) => tab.condition ?? true);

  return (
    <Flex bg="bg" color="fg" direction="column" minH="100dvh">
      <HomeTopBar />
      <Box css={HOME_SCREEN_CONTENT_WRAPPER_SX}>
        <HomeTabs tabs={tabs} />
      </Box>
    </Flex>
  );
};

const HomeTabs = ({ tabs }: { tabs: HomeTabDefinition[] }) => {
  const location = useLocation();
  const requestedTabId = getRequestedHomeTabId(location.pathname, location.hash);
  const activeTabId = getActiveHomeTabId(tabs, requestedTabId);

  useEffect(() => {
    if (window.location.hash !== `#${activeTabId}`) {
      replaceHomeTabHash(activeTabId);
    }
  }, [activeTabId]);

  return (
    <Tabs.Root
      display="flex"
      flexDirection={{ base: 'column', md: 'row' }}
      gap={{ base: '6', md: '10' }}
      minH="0"
      orientation="vertical"
      value={activeTabId}
      variant="subtle"
      onValueChange={(details) => {
        if (isHomeTabId(details.value)) {
          pushHomeTabHash(details.value);
        }
      }}
    >
      <Stack flexShrink={0} gap="4" w={{ base: 'full', md: '60' }}>
        {tabs.length > 1 ? (
          <Tabs.List>
            {tabs.map((tab) => (
              <Tabs.Trigger key={tab.id} value={tab.id}>
                <Icon as={tab.icon} boxSize="3.5" flexShrink={0} />
                <Text truncate>{tab.label}</Text>
              </Tabs.Trigger>
            ))}
          </Tabs.List>
        ) : null}

        <ResourceLinks />
      </Stack>
      <Box flex="1" minW="0">
        {tabs.map((tab) => (
          <Tabs.Content key={tab.id} m="0" p="0" value={tab.id}>
            {tab.children}
          </Tabs.Content>
        ))}
      </Box>
    </Tabs.Root>
  );
};

const UsersPanel = () => (
  <Box flex="1" minW={0}>
    <UsersManagementPanel />
  </Box>
);

const ProjectsPanel = ({ greeting, onImport }: { greeting: string; onImport: () => Promise<void> }) => (
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
      <HStack gap="2" wrap="wrap">
        <Button size="xs" variant="outline" onClick={() => void onImport()}>
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
);
