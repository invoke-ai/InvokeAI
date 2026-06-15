import { Alert, SimpleGrid, Skeleton, Stack } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { refreshProjectLibrary, useProjectLibrary } from '@workbench/projects/library';

import { NewProjectCard } from './NewProjectCard';
import { ProjectCard } from './ProjectCard';

/**
 * The saved-projects grid, fed by the project library store (summaries are
 * already sorted most-recently-edited first). The "new project" cell always
 * leads, so an empty library still presents the next step.
 */
export const ProjectsGrid = () => {
  const library = useProjectLibrary();
  const isFirstLoad = library.summaries.length === 0 && (library.status === 'idle' || library.status === 'loading');

  return (
    <Stack gap="3">
      {library.status === 'error' ? (
        <Alert.Root borderRadius="md" size="sm" status="error">
          <Alert.Indicator />
          <Alert.Title flex="1" fontSize="xs">
            {library.error ?? 'Failed to load your projects.'}
          </Alert.Title>
          <Button size="2xs" variant="outline" onClick={() => void refreshProjectLibrary()}>
            Retry
          </Button>
        </Alert.Root>
      ) : null}
      <SimpleGrid columns={{ base: 1, lg: 3, sm: 2 }} gap="4">
        <NewProjectCard />
        {isFirstLoad
          ? Array.from({ length: 3 }, (_value, index) => <Skeleton key={index} minH="40" rounded="lg" />)
          : library.summaries.map((summary) => <ProjectCard key={summary.id} summary={summary} />)}
      </SimpleGrid>
    </Stack>
  );
};
