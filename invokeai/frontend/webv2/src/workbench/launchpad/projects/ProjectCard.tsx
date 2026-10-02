import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { Box, Flex, Icon, Stack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Link } from '@tanstack/react-router';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { isProjectSummaryCompatible } from '@workbench/projects/library';
import { EllipsisVerticalIcon, PinIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useProjectActionsMenu, useProjectActionsMenuTrigger } from './ProjectActionsMenuHost';
import { ProjectCompatibilityBadge } from './ProjectCompatibilityBadge';
import { ProjectCover } from './ProjectCover';

/**
 * Cards deep-link to the editor and preload on hover; the shared menu host handles overflow and right-click
 * actions.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const REVEAL_ON_HOVER = { opacity: 1 } as const;
const LINK_STYLE = { cursor: 'default', inset: 0, position: 'absolute' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';

export const ProjectCard = ({
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
  const isCompatible = isProjectSummaryCompatible(summary);

  const projectSearch = useMemo(() => ({ project: summary.id }), [summary.id]);
  const menuTarget = useMemo(() => ({ isPinned, onTogglePin, summary }), [isPinned, onTogglePin, summary]);
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLDivElement>) => menu.openAtPointer(event, menuTarget),
    [menu, menuTarget]
  );
  const menuTrigger = useProjectActionsMenuTrigger(menuTarget);
  const handleTogglePin = useCallback(() => onTogglePin(summary.id), [onTogglePin, summary.id]);

  return (
    <Box
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      className="group"
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
          search={projectSearch}
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
      <Box pointerEvents="none">
        <ProjectCover coverUrl={summary.coverUrl} />
      </Box>
      <Flex align="center" gap="2" p="3" pointerEvents="none">
        <Stack flex="1" gap="0" minW="0">
          <MiddleTruncate fontSize="xs" fontWeight="600" text={summary.name} />
          <Text color="fg.muted" fontSize="2xs">
            {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
          </Text>
          <ProjectCompatibilityBadge summary={summary} />
        </Stack>
      </Flex>

      <Box left="2" pointerEvents="auto" position="absolute" top="2" zIndex="1">
        <IconButton
          aria-label={isPinned ? t('projects.unpin') : t('projects.pin')}
          aria-pressed={isPinned}
          color={isPinned ? 'fg' : 'fg.muted'}
          opacity={isPinned ? 1 : 0}
          size="2xs"
          title={isPinned ? t('projects.unpin') : t('projects.pin')}
          variant="subtle"
          _focusVisible={REVEAL_ON_HOVER}
          _groupHover={REVEAL_ON_HOVER}
          onClick={handleTogglePin}
        >
          <Icon as={PinIcon} boxSize="3" fill={isPinned ? 'currentColor' : 'none'} />
        </IconButton>
      </Box>

      <Box bottom="2" pointerEvents="auto" position="absolute" right="2" zIndex="1">
        <IconButton
          aria-expanded={menuTrigger.isExpanded}
          aria-haspopup="menu"
          aria-label={t('common.actions')}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={menuTrigger.onClick}
          onPointerDown={menuTrigger.onPointerDown}
        >
          <EllipsisVerticalIcon />
        </IconButton>
      </Box>
    </Box>
  );
};
