/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import { Skeleton, Stack, Text } from '@chakra-ui/react';
import { useAuthSession, useCapabilities } from '@features/identity';
import { LAUNCHPAD_READY_MARK, markSemanticReady } from '@platform/performance/semanticReady';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, toaster } from '@platform/ui';
import { PageShell } from '@platform/ui/PageShell';
import { Link, useNavigate } from '@tanstack/react-router';
import { IntentTiles } from '@workbench/launchpad/home/IntentTiles';
import { LivePanel } from '@workbench/launchpad/home/LivePanel';
import { RecentProjectsRow } from '@workbench/launchpad/home/RecentProjectsRow';
import { ResumeCard } from '@workbench/launchpad/home/ResumeCard';
import { KnownBrowserIssuesAlert } from '@workbench/launchpad/KnownBrowserIssuesAlert';
import { prunePinnedProjects, toggleProjectPinPreference } from '@workbench/launchpad/projects/projectPins';
import { getProjectLibrary, refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
import { refreshOpenProjects } from '@workbench/projects/openProjects';
import { importProjectFile, pickProjectFile } from '@workbench/projects/projectFile';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { FileUpIcon, PlusIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The Launchpad's landing section: resume what you were doing, or start
 * something new.
 *
 * Home deliberately shows only a handful of recent projects — the full library
 * is its own section. What it adds over a file list is the first move: intent
 * tiles that open a draft already arranged for the kind of work you named.
 */

const NEW_PROJECT_SEARCH = { new: true } as const;
const RECENT_PROJECT_COUNT = 4;
const BROWSER_ISSUES_BANNER = <KnownBrowserIssuesAlert />;

export const HomePage = () => {
  const session = useAuthSession();
  const { canManageModels } = useCapabilities();
  const navigate = useNavigate();
  const { t } = useTranslation();

  const status = useProjectLibrarySelector((snapshot) => snapshot.status);
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const pinnedIds = useWorkbenchPreferenceSelector((preferences) => preferences.launchpadPinnedProjectIds);

  useMountEffect(() => {
    const owner = captureAccountScope();

    void refreshOpenProjects();
    void refreshProjectLibrary().then(() => {
      if (isAccountScopeCurrent(owner)) {
        prunePinnedProjects(getProjectLibrary().summaries);
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

  const isFirstLoad = summaries.length === 0 && (status === 'idle' || status === 'loading');
  const [mostRecent, ...rest] = summaries;

  return (
    <PageShell
      actions={
        <>
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
        </>
      }
      banner={BROWSER_ISSUES_BANNER}
      description={t('launchpad.projectsSubtitle')}
      regionLabel={t('launchpad.sections.home')}
      title={greeting}
    >
      {/* Ordered by how much it should interrupt: a missing model blocks
          everything, a running job is worth knowing about, the rest is
          browsing. Each renders nothing when it has nothing to say.

          The models panel is capability-gated rather than merely hidden: every
          endpoint it touches — catalog, starters, install — is admin-only, so
          mounting it for a non-admin would mean a burst of unauthorized
          requests on every visit to Home, and a call to action they could not
          complete. */}
      {canManageModels ? <LivePanel panel="models" /> : null}
      <LivePanel panel="queue" />

      {isFirstLoad ? <Skeleton minH="24" rounded="lg" /> : mostRecent ? <ResumeCard summary={mostRecent} /> : null}

      <Stack gap="3">
        <Text fontSize="xs" fontWeight="700">
          {t('launchpad.home.intents.heading')}
        </Text>
        <IntentTiles />
      </Stack>

      <RecentProjectsRow
        pinnedIds={pinnedIds}
        summaries={rest.slice(0, RECENT_PROJECT_COUNT)}
        onTogglePin={toggleProjectPinPreference}
      />

      <LivePanel panel="outputs" />
    </PageShell>
  );
};
