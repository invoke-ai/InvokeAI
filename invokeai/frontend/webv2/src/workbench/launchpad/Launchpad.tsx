import { Box, Flex } from '@chakra-ui/react';
import { useLocation, useNavigate } from '@tanstack/react-router';
import { useCapabilities } from '@workbench/auth/capabilities';
import { Tabs } from '@workbench/components/ui';
import { BoxIcon, BlocksIcon, FolderIcon, UsersIcon, type LucideIcon } from 'lucide-react';
import { type ReactNode } from 'react';

import { LaunchpadNav } from './LaunchpadNav';
import { LaunchpadTopBar } from './LaunchpadTopBar';
import { ModelsPage } from './pages/ModelsPage';
import { NodesPage } from './pages/NodesPage';
import { ProjectsPage } from './pages/ProjectsPage';
import { UsersPage } from './pages/UsersPage';

/**
 * The landing surface at `/`: a full-height shell with a slim section rail and
 * a full-width page area — the editor equivalent of Photoshop's home screen.
 * Sections live as switchable pages (Projects, the model manager, and admin
 * user management) so the surface has room to grow without crowding any one of
 * them. It deliberately mounts none of the workbench providers or runtimes, so
 * it stays on the light side of the route-level code split; the heaviest page,
 * the model manager, lazy-loads its own chunk the first time it is opened.
 */

type LaunchpadSectionId = 'projects' | 'models' | 'nodes' | 'users';

interface LaunchpadSection {
  id: LaunchpadSectionId;
  label: string;
  icon: LucideIcon;
  render: () => ReactNode;
  condition?: boolean;
}

const DEFAULT_SECTION_ID: LaunchpadSectionId = 'projects';
const SECTION_IDS: readonly string[] = ['projects', 'models', 'nodes', 'users'];

const isSectionId = (value: string): value is LaunchpadSectionId => SECTION_IDS.includes(value);

const normalizeSectionId = (value: string): LaunchpadSectionId | null => {
  const id = value.replace(/^\/+/, '');

  return isSectionId(id) ? id : null;
};

const SECTION_PATHS: Record<LaunchpadSectionId, '/projects' | '/models' | '/nodes' | '/users'> = {
  models: '/models',
  nodes: '/nodes',
  projects: '/projects',
  users: '/users',
};

const getRequestedSectionId = (pathname: string): LaunchpadSectionId | null => normalizeSectionId(pathname);

const getActiveSectionId = (
  sections: LaunchpadSection[],
  requestedSectionId: LaunchpadSectionId | null
): LaunchpadSectionId =>
  requestedSectionId && sections.some((section) => section.id === requestedSectionId)
    ? requestedSectionId
    : DEFAULT_SECTION_ID;

export const Launchpad = () => {
  const { canManageModels, canManageNodes, canManageUsers } = useCapabilities();

  const allSections: LaunchpadSection[] = [
    { icon: FolderIcon, id: 'projects', label: 'Projects', render: () => <ProjectsPage /> },
    { condition: canManageModels, icon: BoxIcon, id: 'models', label: 'Models', render: () => <ModelsPage /> },
    { condition: canManageNodes, icon: BlocksIcon, id: 'nodes', label: 'Nodes', render: () => <NodesPage /> },
    { condition: canManageUsers, icon: UsersIcon, id: 'users', label: 'Users', render: () => <UsersPage /> },
  ];
  const sections = allSections.filter((section) => section.condition ?? true);

  return (
    <Flex bg="bg" color="fg" direction="column" h="100dvh" overflow="hidden">
      <LaunchpadTopBar />
      <LaunchpadSections sections={sections} />
    </Flex>
  );
};

const LaunchpadSections = ({ sections }: { sections: LaunchpadSection[] }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const requestedSectionId = getRequestedSectionId(location.pathname);
  const activeSectionId = getActiveSectionId(sections, requestedSectionId);

  return (
    <Tabs.Root
      display="flex"
      flex="1"
      flexDirection={{ base: 'column', md: 'row' }}
      lazyMount
      minH="0"
      size="sm"
      orientation="vertical"
      value={activeSectionId}
      variant="subtle"
      onValueChange={(details) => {
        if (isSectionId(details.value)) {
          void navigate({ to: SECTION_PATHS[details.value] });
        }
      }}
    >
      <LaunchpadNav items={sections} />
      <Box flex="1" minH="0" minW="0" position="relative">
        {sections.map((section) => (
          <Tabs.Content key={section.id} h="full" m="0" minH="0" p="0" value={section.id}>
            {section.render()}
          </Tabs.Content>
        ))}
      </Box>
    </Tabs.Root>
  );
};
