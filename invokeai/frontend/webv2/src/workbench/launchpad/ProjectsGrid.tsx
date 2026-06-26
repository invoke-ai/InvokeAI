import { Alert, SimpleGrid, Skeleton, Stack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
import { useCallback } from 'react';

import { NewProjectCard } from './NewProjectCard';
import { ProjectCard } from './ProjectCard';

const GRID_COLUMNS = { base: 1, lg: 3, sm: 2 } as const;

/**
 * The saved-projects grid, fed by the project library store (summaries are
 * already sorted most-recently-edited first). The "new project" cell always
 * leads, so an empty library still presents the next step.
 */
export const ProjectsGrid = () => {
  const error = useProjectLibrarySelector((snapshot) => snapshot.error);
  const status = useProjectLibrarySelector((snapshot) => snapshot.status);
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);
  const isFirstLoad = summaries.length === 0 && (status === 'idle' || status === 'loading');
  const handleRetry = useCallback(() => void refreshProjectLibrary(), []);

  return (
    <Stack gap="3">
      {status === 'error' ? (
        <Alert.Root borderRadius="md" size="sm" status="error">
          <Alert.Indicator />
          <Alert.Title flex="1" fontSize="xs">
            {error ?? 'Failed to load your projects.'}
          </Alert.Title>
          <Button size="2xs" variant="outline" onClick={handleRetry}>
            Retry
          </Button>
        </Alert.Root>
      ) : null}
      <SimpleGrid columns={GRID_COLUMNS} gap="4">
        <NewProjectCard />
        {isFirstLoad
          ? Array.from({ length: 3 }, (_value, index) => <Skeleton key={index} minH="40" rounded="lg" />)
          : summaries.map((summary) => <ProjectCard key={summary.id} summary={summary} />)}
      </SimpleGrid>
    </Stack>
  );
};
