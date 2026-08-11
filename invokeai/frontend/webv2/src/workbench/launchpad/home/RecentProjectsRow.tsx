import type { ProjectSummary } from '@workbench/projects/library';

import { Flex, SimpleGrid, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Link } from '@tanstack/react-router';
import { ProjectCard } from '@workbench/launchpad/projects/ProjectCard';
import { ArrowRightIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * A short strip of recent projects under the resume card — enough to recognise
 * what else is in flight without turning Home into a second library. The full
 * one is a click away.
 */

const GRID_COLUMNS = { base: 1, lg: 4, sm: 2 } as const;

export const RecentProjectsRow = ({
  pinnedIds,
  summaries,
  onTogglePin,
}: {
  pinnedIds: readonly string[];
  summaries: readonly ProjectSummary[];
  onTogglePin: (projectId: string) => void;
}) => {
  const { t } = useTranslation();

  if (summaries.length === 0) {
    return null;
  }

  return (
    <Flex direction="column" gap="3">
      <Flex align="center" justify="space-between">
        <Text fontSize="xs" fontWeight="700">
          {t('launchpad.home.recentProjects')}
        </Text>
        <Button asChild size="2xs" variant="ghost">
          <Link to="/projects">
            {t('launchpad.home.viewAllProjects')}
            <ArrowRightIcon />
          </Link>
        </Button>
      </Flex>
      <SimpleGrid columns={GRID_COLUMNS} gap="4">
        {summaries.map((summary) => (
          <ProjectCard
            isPinned={pinnedIds.includes(summary.id)}
            key={summary.id}
            summary={summary}
            onTogglePin={onTogglePin}
          />
        ))}
      </SimpleGrid>
    </Flex>
  );
};
