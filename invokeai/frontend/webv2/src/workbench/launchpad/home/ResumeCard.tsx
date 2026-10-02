import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { Box, Flex, HStack, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Link } from '@tanstack/react-router';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import {
  useProjectActionsMenu,
  useProjectActionsMenuTrigger,
} from '@workbench/launchpad/projects/ProjectActionsMenuHost';
import { ProjectCompatibilityBadge } from '@workbench/launchpad/projects/ProjectCompatibilityBadge';
import { ProjectCover } from '@workbench/launchpad/projects/ProjectCover';
import { isProjectSummaryCompatible } from '@workbench/projects/library';
import { ArrowRightIcon, EllipsisVerticalIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const LINK_STYLE = { cursor: 'default', inset: 0, position: 'absolute' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';
const COVER_WIDTH = { base: '32', sm: '40' } as const;

export const ResumeCard = ({
  isPinned,
  summary,
  onTogglePin,
}: {
  isPinned: boolean;
  summary: ProjectSummary;
  onTogglePin: (projectId: string) => void;
}) => {
  const { t } = useTranslation();
  const menu = useProjectActionsMenu();
  const search = useMemo(() => ({ project: summary.id }), [summary.id]);
  const isCompatible = isProjectSummaryCompatible(summary);

  const menuTarget = useMemo(() => ({ isPinned, onTogglePin, summary }), [isPinned, onTogglePin, summary]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => menu.openAtPointer(event, menuTarget),
    [menu, menuTarget]
  );
  const menuTrigger = useProjectActionsMenuTrigger(menuTarget);

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
      onContextMenu={handleContextMenu}
    >
      {isCompatible ? (
        <Link
          aria-label={t('projects.openProjectLabel', { name: summary.name })}
          search={search}
          style={LINK_STYLE}
          to="/app"
        />
      ) : (
        <Box
          aria-disabled="true"
          aria-label={`${t('projects.openProjectLabel', { name: summary.name })}. ${t('projects.file.updateClient')}`}
          role="link"
          style={LINK_STYLE}
          tabIndex={0}
          title={t('projects.file.updateClient')}
        />
      )}
      <Flex align="center" flexShrink={0} p="1.5" pointerEvents="none" w={COVER_WIDTH}>
        <Box overflow="hidden" rounded="md" w="full">
          <ProjectCover coverUrl={summary.coverUrl} />
        </Box>
      </Flex>
      <Flex align="center" flex="1" gap="3" justify="space-between" minW="0" p="4" pointerEvents="none" wrap="wrap">
        <Stack gap="0.5" minW="0">
          <Text color="fg.muted" fontSize="2xs" fontWeight="600" textTransform="uppercase">
            {t('launchpad.home.resume')}
          </Text>
          <MiddleTruncate fontSize="sm" fontWeight="700" text={summary.name} />
          <Text color="fg.muted" fontSize="2xs">
            {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
          </Text>
          <ProjectCompatibilityBadge summary={summary} />
        </Stack>
        <HStack gap="1.5" pointerEvents="auto">
          {isCompatible ? (
            <Button asChild size="xs" variant="solid">
              <Link search={search} to="/app">
                {t('launchpad.home.resumeAction')}
                <ArrowRightIcon />
              </Link>
            </Button>
          ) : (
            <Button disabled size="xs" title={t('projects.file.updateClient')} variant="solid">
              {t('launchpad.home.resumeAction')}
              <ArrowRightIcon />
            </Button>
          )}
          <IconButton
            aria-expanded={menuTrigger.isExpanded}
            aria-haspopup="menu"
            aria-label={t('common.actions')}
            color="fg.muted"
            size="xs"
            variant="ghost"
            onClick={menuTrigger.onClick}
            onPointerDown={menuTrigger.onPointerDown}
          >
            <EllipsisVerticalIcon />
          </IconButton>
        </HStack>
      </Flex>
    </Flex>
  );
};
