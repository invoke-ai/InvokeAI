import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { Box, Flex, Icon, Menu, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Row } from '@platform/ui/Row';
import { Link } from '@tanstack/react-router';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { EllipsisVerticalIcon, PinIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ProjectActionsMenu } from './ProjectActionsMenu';
import { ProjectCover } from './ProjectCover';
import { useProjectCardActions } from './useProjectCardActions';

/**
 * One saved project in the list view — the density a large library wants.
 * Carries the same link target and the same action menu as the grid card;
 * only the shape differs.
 */

const LINK_STYLE = { inset: 0, position: 'absolute' } as const;
const REVEAL_ON_HOVER = { opacity: 1 } as const;
const THUMBNAIL_WIDTH = '14';

export const ProjectRow = ({
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
    <Row className="group" gap="3" h="full" position="relative" rounded="md" onContextMenu={handleContextMenu}>
      <Link
        aria-label={t('projects.openProjectLabel', { name: summary.name })}
        search={projectSearch}
        style={LINK_STYLE}
        to="/app"
      />
      <Box flexShrink={0} overflow="hidden" pointerEvents="none" rounded="sm" w={THUMBNAIL_WIDTH}>
        <ProjectCover coverUrl={summary.coverUrl} />
      </Box>
      <Flex flex="1" gap="3" minW="0" pointerEvents="none">
        <MiddleTruncate flex="1" fontSize="xs" fontWeight="600" minW="0" text={summary.name} />
        <Text color="fg.muted" flexShrink={0} fontSize="2xs">
          {t('projects.editedRelative', { time: formatRelativeTime(summary.updatedAt) })}
        </Text>
      </Flex>
      <Flex flexShrink={0} gap="0.5" pointerEvents="auto" zIndex="1">
        <IconButton
          aria-label={isPinned ? t('projects.unpin') : t('projects.pin')}
          aria-pressed={isPinned}
          color={isPinned ? 'fg' : 'fg.muted'}
          opacity={isPinned ? 1 : 0}
          size="2xs"
          title={isPinned ? t('projects.unpin') : t('projects.pin')}
          variant="ghost"
          _focusVisible={REVEAL_ON_HOVER}
          _groupHover={REVEAL_ON_HOVER}
          onClick={handleTogglePin}
        >
          <Icon as={PinIcon} boxSize="3" fill={isPinned ? 'currentColor' : 'none'} />
        </IconButton>
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
      </Flex>
    </Row>
  );
};
