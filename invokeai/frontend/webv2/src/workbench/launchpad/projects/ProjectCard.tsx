import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { Box, Flex, Icon, Menu, Stack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Link } from '@tanstack/react-router';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { EllipsisVerticalIcon, PinIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ProjectActionsMenu } from './ProjectActionsMenu';
import { ProjectCover } from './ProjectCover';
import { useProjectCardActions } from './useProjectCardActions';

/**
 * One saved project in the library grid. The whole card is a deep link into
 * the editor (`/app?project=…` — hovering preloads the editor chunk); the
 * corner menu and right-click carry the library actions.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const REVEAL_ON_HOVER = { opacity: 1 } as const;
const LINK_STYLE = { inset: 0, position: 'absolute' } as const;
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
  const actions = useProjectCardActions(summary);
  const [isActionsOpen, setIsActionsOpen] = useState(false);
  const [contextMenuTarget, setContextMenuTarget] = useState<{ x: number; y: number } | null>(null);

  const projectSearch = useMemo(() => ({ project: summary.id }), [summary.id]);
  const handleContextMenu = useCallback((event: MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    setContextMenuTarget({ x: event.clientX, y: event.clientY });
    setIsActionsOpen(true);
  }, []);
  const handleOpenChange = useCallback((event: { open: boolean }) => {
    setIsActionsOpen(event.open);

    if (!event.open) {
      setContextMenuTarget(null);
    }
  }, []);
  const clearContextMenuTarget = useCallback(() => setContextMenuTarget(null), []);
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
      <Link
        aria-label={t('projects.openProjectLabel', { name: summary.name })}
        search={projectSearch}
        style={LINK_STYLE}
        to="/app"
      />
      <Box pointerEvents="none">
        <ProjectCover coverUrl={summary.coverUrl} />
      </Box>
      <Flex align="center" gap="2" p="3" pointerEvents="none">
        <Stack flex="1" gap="0" minW="0">
          <MiddleTruncate fontSize="xs" fontWeight="600" text={summary.name} />
          <Text color="fg.muted" fontSize="2xs">
            {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
          </Text>
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
        <ProjectActionsMenu
          actions={actions}
          contextMenuTarget={contextMenuTarget}
          isOpen={isActionsOpen}
          isPinned={isPinned}
          projectId={summary.id}
          projectName={summary.name}
          onOpenChange={handleOpenChange}
          onTogglePin={handleTogglePin}
        >
          <Menu.Trigger asChild>
            <IconButton
              aria-label={t('common.actions')}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={clearContextMenuTarget}
            >
              <EllipsisVerticalIcon />
            </IconButton>
          </Menu.Trigger>
        </ProjectActionsMenu>
      </Box>
    </Box>
  );
};
