import type { ProjectSummary } from '@workbench/projects/library';

import { Box, Flex, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Link } from '@tanstack/react-router';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { ProjectCover } from '@workbench/launchpad/projects/ProjectCover';
import { ArrowRightIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The single most recently edited project, given more room than the rest.
 *
 * Returning users almost always want the thing they were last working on, and
 * making them find it in a grid of equals is the small tax the old home screen
 * charged on every visit.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const LINK_STYLE = { inset: 0, position: 'absolute' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';
const COVER_WIDTH = { base: '32', sm: '40' } as const;

export const ResumeCard = ({ summary }: { summary: ProjectSummary }) => {
  const { t } = useTranslation();
  const search = useMemo(() => ({ project: summary.id }), [summary.id]);

  return (
    <Flex
      align="stretch"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      overflow="hidden"
      position="relative"
      rounded="lg"
      transition={CARD_TRANSITION}
      _hover={CARD_HOVER}
    >
      <Link
        aria-label={t('projects.openProjectLabel', { name: summary.name })}
        search={search}
        style={LINK_STYLE}
        to="/app"
      />
      <Box flexShrink={0} pointerEvents="none" w={COVER_WIDTH}>
        <ProjectCover coverUrl={summary.coverUrl} />
      </Box>
      <Flex align="center" flex="1" gap="3" justify="space-between" minW="0" p="4" pointerEvents="none" wrap="wrap">
        <Stack gap="0.5" minW="0">
          <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
            {t('launchpad.home.resume')}
          </Text>
          <MiddleTruncate fontSize="sm" fontWeight="700" text={summary.name} />
          <Text color="fg.muted" fontSize="2xs">
            {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
          </Text>
        </Stack>
        <Box pointerEvents="auto">
          <Button asChild size="xs" variant="solid">
            <Link search={search} to="/app">
              {t('launchpad.home.resumeAction')}
              <ArrowRightIcon />
            </Link>
          </Button>
        </Box>
      </Flex>
    </Flex>
  );
};
