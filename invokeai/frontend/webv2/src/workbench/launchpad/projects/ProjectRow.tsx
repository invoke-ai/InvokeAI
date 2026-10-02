import type { ProjectSummary } from '@workbench/projects/library';
import type { MouseEvent } from 'react';

import { Box, Flex, Icon, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Row } from '@platform/ui/Row';
import { Link } from '@tanstack/react-router';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { isProjectSummaryCompatible } from '@workbench/projects/library';
import { EllipsisVerticalIcon, PinIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useProjectActionsMenu, useProjectActionsMenuTrigger } from './ProjectActionsMenuHost';
import { ProjectCompatibilityBadge } from './ProjectCompatibilityBadge';
import { ProjectCover } from './ProjectCover';

const LINK_STYLE = { cursor: 'default', inset: 0, position: 'absolute' } as const;
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
    <Row className="group" gap="3" h="full" position="relative" px="2.5" rounded="md" onContextMenu={handleContextMenu}>
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
      <Box flexShrink={0} overflow="hidden" pointerEvents="none" rounded="sm" w={THUMBNAIL_WIDTH}>
        <ProjectCover coverUrl={summary.coverUrl} />
      </Box>
      <Flex flex="1" gap="3" minW="0" pointerEvents="none">
        <MiddleTruncate flex="1" fontSize="xs" fontWeight="600" minW="0" text={summary.name} />
        <ProjectCompatibilityBadge summary={summary} />
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
      </Flex>
    </Row>
  );
};
