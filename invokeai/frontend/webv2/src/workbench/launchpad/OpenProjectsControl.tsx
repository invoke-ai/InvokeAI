import type { SystemStyleObject } from '@chakra-ui/react';

import { Badge, Icon, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { Tooltip } from '@platform/ui/Tooltip';
import { Link } from '@tanstack/react-router';
import { useProjectLibrarySelector } from '@workbench/projects/library';
import { refreshOpenProjects, useOpenProjectsSelector } from '@workbench/projects/openProjects';
import { AppWindowIcon, ArrowUpRightIcon, FolderOpenIcon } from 'lucide-react';
import { useId, useMemo, type ComponentProps } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Hide editor re-entry only for a known-empty saved session. Unknown sessions retain entry so the editor can
 * resolve them.
 */

/** Many open projects scroll inside the section instead of pushing Manage out of the rail. */
const LIST_MAX_H = '48';

/** These entries leave the Launchpad for the editor; a corner arrow says so on hover and keyboard focus. */
const LEAVE_ENTRY_CSS: SystemStyleObject = {
  '& [data-leave-icon]': {
    opacity: 0,
    transitionDuration: 'var(--wb-motion-duration-fast)',
    transitionProperty: 'opacity',
  },
  '&:hover [data-leave-icon], &:focus-visible [data-leave-icon]': { opacity: 1 },
};

type NavItemProps = Pick<ComponentProps<typeof Button>, 'justifyContent' | 'size' | 'w'>;

/** `compact` is the icon-only rail: each entry is named by a tooltip and the heading is read, not shown. */
export const OpenProjectsNavSection = ({
  compact = false,
  headingCss,
  itemProps,
}: {
  compact?: boolean;
  headingCss: SystemStyleObject;
  itemProps: NavItemProps;
}) => {
  const { t } = useTranslation();
  const headingId = useId();
  const status = useOpenProjectsSelector((snapshot) => snapshot.status);
  const openProjectIds = useOpenProjectsSelector((snapshot) => snapshot.ids);
  const activeProjectId = useOpenProjectsSelector((snapshot) => snapshot.activeId);
  const summaries = useProjectLibrarySelector((snapshot) => snapshot.summaries);

  useMountEffect(() => {
    void refreshOpenProjects();
  });

  const openProjects = useMemo(() => {
    if (!openProjectIds) {
      return [];
    }

    const nameById = new Map(summaries.map((summary) => [summary.id, summary.name]));
    // The current project leads; the rest keep their open order.
    const ordered =
      activeProjectId && openProjectIds.includes(activeProjectId)
        ? [activeProjectId, ...openProjectIds.filter((id) => id !== activeProjectId)]
        : openProjectIds;

    return ordered.map((id) => ({ id, name: nameById.get(id) ?? null }));
  }, [activeProjectId, openProjectIds, summaries]);

  if (status !== 'ready' || (openProjectIds !== null && openProjects.length === 0)) {
    return null;
  }

  return (
    <Stack aria-labelledby={headingId} as="section" flexShrink={1} gap="0.5" minH="0">
      {compact ? (
        <VisuallyHidden id={headingId}>{t('launchpad.openProjects.label')}</VisuallyHidden>
      ) : (
        <Text css={headingCss} id={headingId}>
          {t('launchpad.openProjects.label')}
        </Text>
      )}
      <Stack gap="0.5" maxH={LIST_MAX_H} minH="0" overflowY="auto">
        {compact ? (
          openProjectIds === null ? (
            <CompactEntry label={t('launchpad.openProjects.openEditor')} />
          ) : (
            openProjects.map((project) => (
              <CompactEntry
                key={project.id}
                isCurrent={project.id === activeProjectId}
                label={
                  project.id === activeProjectId
                    ? `${project.name ?? project.id} · ${t('launchpad.openProjects.current')}`
                    : (project.name ?? project.id)
                }
                projectId={project.id}
              />
            ))
          )
        ) : openProjectIds === null ? (
          <Button asChild {...itemProps} css={LEAVE_ENTRY_CSS} variant="ghost">
            <Link to="/app">
              <Icon as={FolderOpenIcon} boxSize="3.5" flexShrink={0} />
              <Text flex="1" minW="0" textAlign="start" truncate>
                {t('launchpad.openProjects.openEditor')}
              </Text>
              <LeaveIcon />
            </Link>
          </Button>
        ) : (
          openProjects.map((project) => (
            <OpenProjectEntry
              key={project.id}
              id={project.id}
              isActive={project.id === activeProjectId}
              itemProps={itemProps}
              name={project.name}
            />
          ))
        )}
      </Stack>
    </Stack>
  );
};

/**
 * An icon-rail entry into the editor: an app window rather than the Projects folder, so the two read apart. The
 * current project takes the rail's selected tone; its tooltip and name say "Current" too.
 */
const CompactEntry = ({
  isCurrent = false,
  label,
  projectId,
}: {
  isCurrent?: boolean;
  label: string;
  projectId?: string;
}) => {
  const search = useMemo(() => (projectId === undefined ? undefined : { project: projectId }), [projectId]);

  return (
    <Tooltip content={label} placement="right">
      <IconButton
        asChild
        aria-label={label}
        data-current-project={isCurrent ? '' : undefined}
        size="lg"
        variant={isCurrent ? 'subtle' : 'ghost'}
      >
        <Link search={search} to="/app">
          <Icon as={AppWindowIcon} boxSize="3.5" />
        </Link>
      </IconButton>
    </Tooltip>
  );
};

const LeaveIcon = () => <Icon as={ArrowUpRightIcon} aria-hidden boxSize="3.5" data-leave-icon flexShrink={0} />;

const OpenProjectEntry = ({
  id,
  isActive,
  itemProps,
  name,
}: {
  id: string;
  isActive: boolean;
  itemProps: NavItemProps;
  name: string | null;
}) => {
  const { t } = useTranslation();
  const search = useMemo(() => ({ project: id }), [id]);

  return (
    <Button asChild {...itemProps} css={LEAVE_ENTRY_CSS} variant="ghost">
      <Link search={search} to="/app">
        <Icon as={FolderOpenIcon} boxSize="3.5" flexShrink={0} />
        <MiddleTruncate flex="1" minW="0" text={name ?? id} textAlign="start" />
        {isActive ? (
          <Badge flexShrink={0} variant="subtle">
            {t('launchpad.openProjects.current')}
          </Badge>
        ) : null}
        <LeaveIcon />
      </Link>
    </Button>
  );
};
