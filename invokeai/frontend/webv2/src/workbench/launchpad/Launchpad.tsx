import { Box, Flex, VisuallyHidden, type SystemStyleObject } from '@chakra-ui/react';
import { FontsPage } from '@features/fonts/launchpad';
import { useCapabilities, UsersPage } from '@features/identity';
import { INTERMEDIATES_SETTING_ID, requestIntermediatesFocus } from '@features/intermediates';
import { ModelsPage } from '@features/models';
import { NodesPage } from '@features/nodes';
import { useLocation, useNavigate } from '@tanstack/react-router';
import { LaunchpadCommandPalette } from '@workbench/palette/LaunchpadCommandPalette';
import { PreferencesPage } from '@workbench/settings/launchpad';
import {
  BlocksIcon,
  BoxIcon,
  FolderIcon,
  HouseIcon,
  SettingsIcon,
  TypeIcon,
  UsersIcon,
  type LucideIcon,
} from 'lucide-react';
import { useCallback, useMemo, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { LaunchpadNav, type LaunchpadNavGroupId } from './LaunchpadNav';
import { LaunchpadTopBar } from './LaunchpadTopBar';
import { HomePage } from './pages/HomePage';
import { ProjectsPage } from './pages/ProjectsPage';
import { ProjectActionsMenuProvider } from './projects/ProjectActionsMenuHost';

/**
 * Launchpad owns grouped home/admin pages without mounting workbench runtimes. Model management loads on demand
 * across the route split.
 */

type LaunchpadSectionId = 'home' | 'projects' | 'models' | 'nodes' | 'users' | 'fonts' | 'preferences';

interface LaunchpadSection {
  id: LaunchpadSectionId;
  label: string;
  icon: LucideIcon;
  group: LaunchpadNavGroupId;
  render: () => ReactNode;
  condition?: boolean;
}

const DEFAULT_SECTION_ID: LaunchpadSectionId = 'home';
const SECTION_IDS: readonly string[] = ['home', 'projects', 'models', 'nodes', 'users', 'fonts', 'preferences'];

const isSectionId = (value: string): value is LaunchpadSectionId => SECTION_IDS.includes(value);

/** `/` is Home; every other section is its own path, which may continue into the section (`/preferences/hotkeys`). */
const normalizeSectionId = (value: string): LaunchpadSectionId | null => {
  const id = value.replace(/^\/+/, '').split('/')[0] ?? '';

  if (id === '') {
    return 'home';
  }

  return isSectionId(id) ? id : null;
};

const SECTION_PATHS: Record<
  LaunchpadSectionId,
  '/' | '/projects' | '/models' | '/nodes' | '/users' | '/fonts' | '/preferences'
> = {
  fonts: '/fonts',
  home: '/',
  models: '/models',
  nodes: '/nodes',
  preferences: '/preferences',
  projects: '/projects',
  users: '/users',
};

const SECTIONS_CSS: SystemStyleObject = {
  display: 'flex',
  flex: 1,
  flexDirection: { base: 'column', md: 'row' },
  minH: 0,
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
  const { t } = useTranslation();
  const navigate = useNavigate();
  const manageIntermediatesOf = useCallback(
    (userId: string, label: string) => {
      requestIntermediatesFocus({ ownerId: userId, ownerLabel: label });
      void navigate({
        params: { section: 'intermediates' },
        search: { setting: INTERMEDIATES_SETTING_ID },
        to: '/preferences/$section',
      });
    },
    [navigate]
  );

  const filtered = useMemo<LaunchpadSection[]>(() => {
    const sections = [
      {
        group: 'workspace',
        icon: HouseIcon,
        id: 'home',
        label: t('launchpad.sections.home'),
        render: () => <HomePage />,
      },
      {
        group: 'workspace',
        icon: FolderIcon,
        id: 'projects',
        label: t('launchpad.sections.projects'),
        render: () => <ProjectsPage />,
      },
      {
        condition: canManageModels,
        group: 'manage',
        icon: BoxIcon,
        id: 'models',
        label: t('launchpad.sections.models'),
        render: () => <ModelsPage />,
      },
      {
        condition: canManageNodes,
        group: 'manage',
        icon: BlocksIcon,
        id: 'nodes',
        label: t('launchpad.sections.nodes'),
        render: () => <NodesPage />,
      },
      {
        group: 'manage',
        icon: TypeIcon,
        id: 'fonts',
        label: t('launchpad.sections.fonts'),
        render: () => <FontsPage />,
      },
      {
        condition: canManageUsers,
        group: 'manage',
        icon: UsersIcon,
        id: 'users',
        label: t('launchpad.sections.users'),
        render: () => <UsersPage onManageIntermediates={manageIntermediatesOf} />,
      },
      {
        group: 'footer',
        icon: SettingsIcon,
        id: 'preferences',
        label: t('launchpad.sections.preferences'),
        render: () => <PreferencesPage />,
      },
    ] satisfies (LaunchpadSection & { condition?: boolean })[];

    return sections.filter((section) => section.condition ?? true);
  }, [canManageModels, canManageNodes, canManageUsers, manageIntermediatesOf, t]);

  return (
    <ProjectActionsMenuProvider>
      <Flex bg="bg" color="fg" direction="column" h="100dvh" overflow="hidden">
        <LaunchpadTopBar />
        <LaunchpadSections sections={filtered} />
        <LaunchpadCommandPalette />
      </Flex>
    </ProjectActionsMenuProvider>
  );
};

const LaunchpadSections = ({ sections }: { sections: LaunchpadSection[] }) => {
  const location = useLocation();
  const requestedSectionId = getRequestedSectionId(location.pathname);
  const activeSectionId = getActiveSectionId(sections, requestedSectionId);
  const activeSectionLabel = sections.find((section) => section.id === activeSectionId)?.label ?? activeSectionId;
  const navItems = useMemo(
    () => sections.map((section) => ({ ...section, to: SECTION_PATHS[section.id] })),
    [sections]
  );
  // Visited pages stay mounted (hidden) so returning to one keeps its state, as the former lazily mounted tabs did.
  const [visitedIds, setVisitedIds] = useState<ReadonlySet<LaunchpadSectionId>>(() => new Set([activeSectionId]));

  if (!visitedIds.has(activeSectionId)) {
    setVisitedIds(new Set(visitedIds).add(activeSectionId));
  }

  return (
    <Flex css={SECTIONS_CSS}>
      <LaunchpadNav activeId={activeSectionId} items={navItems} />
      <Box aria-labelledby="launchpad-page-heading" as="main" flex="1" minH="0" minW="0" position="relative">
        <VisuallyHidden as="h1" id="launchpad-page-heading">
          {activeSectionLabel}
        </VisuallyHidden>
        {sections
          .filter((section) => visitedIds.has(section.id))
          .map((section) => (
            <Box key={section.id} h="full" hidden={section.id !== activeSectionId} minH="0">
              {section.render()}
            </Box>
          ))}
      </Box>
    </Flex>
  );
};
