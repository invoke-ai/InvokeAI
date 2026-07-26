import type { SystemStyleObject } from '@chakra-ui/react';

import { Flex, Heading, HStack, Stack, Text } from '@chakra-ui/react';
import { useAuthSession } from '@features/identity';
import { LAUNCHPAD_READY_MARK, markSemanticReady } from '@platform/performance/semanticReady';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, Scrollable, toaster } from '@platform/ui';
import { Link, useNavigate } from '@tanstack/react-router';
import { KnownBrowserIssuesAlert } from '@workbench/launchpad/KnownBrowserIssuesAlert';
import { ProjectsGrid } from '@workbench/launchpad/ProjectsGrid';
import { refreshProjectLibrary } from '@workbench/projects/library';
import { importProjectFile, pickProjectFile } from '@workbench/projects/projectFile';
import { FileUpIcon, PlusIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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

const NEW_PROJECT_SEARCH = { new: true } as const;

export const ProjectsPage = () => {
  const session = useAuthSession();
  const navigate = useNavigate();
  const { t } = useTranslation();

  useMountEffect(() => {
    const owner = captureAccountScope();

    void refreshProjectLibrary().then(() => {
      if (isAccountScopeCurrent(owner)) {
        markSemanticReady(LAUNCHPAD_READY_MARK);
      }
    });
  });

  const displayName = session.user?.display_name?.trim();
  const greeting = displayName
    ? t('launchpad.projectsGreetingWithName', { name: displayName })
    : t('launchpad.projectsGreeting');

  const handleImport = useCallback(async () => {
    const owner = captureAccountScope();
    const file = await pickProjectFile(owner);

    if (!file || !isAccountScopeCurrent(owner)) {
      return;
    }

    try {
      const record = await importProjectFile(file, owner);

      assertAccountScopeCurrent(owner);
      await navigate({ search: { project: record.project_id }, to: '/app' });
      assertAccountScopeCurrent(owner);
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      toaster.create({
        description: error instanceof Error ? error.message : undefined,
        title: t('projects.importFailed'),
        type: 'error',
      });
    }
  }, [navigate, t]);
  const handleImportClick = useCallback(() => void handleImport(), [handleImport]);

  return (
    <Scrollable h="full" label={t('launchpad.sections.projects')} minH="0">
      <Stack css={PROJECTS_PAGE_MEASURE_SX} gap="5">
        <Flex align="center" gap="3" justify="space-between" wrap="wrap">
          <Stack gap="0.5">
            <Heading fontSize="xl" fontWeight="700">
              {greeting}
            </Heading>
            <Text color="fg.muted" fontSize="xs">
              {t('launchpad.projectsSubtitle')}
            </Text>
          </Stack>
          <HStack gap="2" wrap="wrap">
            <Button size="xs" variant="outline" onClick={handleImportClick}>
              <FileUpIcon />
              {t('projects.importWithEllipsis')}
            </Button>
            <Button asChild size="xs" variant="solid">
              <Link search={NEW_PROJECT_SEARCH} to="/app">
                <PlusIcon />
                {t('projects.newProject')}
              </Link>
            </Button>
          </HStack>
        </Flex>
        <KnownBrowserIssuesAlert />
        <ProjectsGrid />
      </Stack>
    </Scrollable>
  );
};
