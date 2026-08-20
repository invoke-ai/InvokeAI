/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';
import type { ProjectRecordDTO } from '@workbench/projects/api';

import { Stack } from '@chakra-ui/react';
import { LAUNCHPAD_READY_MARK, markSemanticReady } from '@platform/performance/semanticReady';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui';
import { PageShell } from '@platform/ui/PageShell';
import { useNavigate } from '@tanstack/react-router';
import { KnownBrowserIssuesAlert } from '@workbench/launchpad/KnownBrowserIssuesAlert';
import { NewProjectButton } from '@workbench/launchpad/projects/NewProjectButton';
import { prunePinnedProjects, toggleProjectPinPreference } from '@workbench/launchpad/projects/projectPins';
import { ProjectsBrowser } from '@workbench/launchpad/projects/ProjectsBrowser';
import { ProjectsToolbar } from '@workbench/launchpad/projects/ProjectsToolbar';
import { getProjectLibrary, refreshProjectLibrary } from '@workbench/projects/library';
import { refreshOpenProjects } from '@workbench/projects/openProjects';
import { useImportProjectFile } from '@workbench/projects/useProjectFileActions';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { FileUpIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The Launchpad's project library.
 *
 * Sort, layout, and pins persist through workbench preferences, so they follow
 * the account rather than the browser. The search term deliberately does not —
 * a filter you did not set is a filter you cannot find your way out of.
 */

export const ProjectsPage = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [searchTerm, setSearchTerm] = useState('');
  // Captured once per visit: the date buckets should reflect when the library
  // was opened, and reading the clock each render would rebucket unstably.
  const [now] = useState(() => Date.now());

  const view = useWorkbenchPreferenceSelector((preferences) => preferences.launchpadProjectsView);
  const sort = useWorkbenchPreferenceSelector((preferences) => preferences.launchpadProjectsSort);
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

  const handleViewChange = useCallback((next: ProjectsViewId) => {
    void patchWorkbenchPreferences({ launchpadProjectsView: next });
  }, []);
  const handleSortChange = useCallback((next: ProjectSortId) => {
    void patchWorkbenchPreferences({ launchpadProjectsSort: next });
  }, []);
  const handleClearSearch = useCallback(() => setSearchTerm(''), []);

  const openImportedProject = useCallback(
    async (record: ProjectRecordDTO) => {
      await navigate({ search: { project: record.project_id }, to: '/app' });
    },
    [navigate]
  );
  const handleImportClick = useImportProjectFile(openImportedProject);

  return (
    <PageShell
      actions={
        <>
          <Button size="xs" variant="outline" onClick={handleImportClick}>
            <FileUpIcon />
            {t('projects.importWithEllipsis')}
          </Button>
          <NewProjectButton />
        </>
      }
      banner={
        <Stack gap="4">
          <KnownBrowserIssuesAlert />
          <ProjectsToolbar
            searchTerm={searchTerm}
            sort={sort}
            view={view}
            onSearchTermChange={setSearchTerm}
            onSortChange={handleSortChange}
            onViewChange={handleViewChange}
          />
        </Stack>
      }
      description={t('projects.libraryDescription')}
      scroll="content"
      title={t('launchpad.sections.projects')}
    >
      <ProjectsBrowser
        now={now}
        pinnedIds={pinnedIds}
        searchTerm={searchTerm}
        sort={sort}
        view={view}
        onClearSearch={handleClearSearch}
        onTogglePin={toggleProjectPinPreference}
      />
    </PageShell>
  );
};
