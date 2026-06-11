import { Icon, Menu, Portal } from '@chakra-ui/react';
import { useRef, useState, type MouseEvent } from 'react';
import { FileDownIcon, FolderCogIcon, FolderOpenIcon, PencilIcon, PlusIcon, Trash2Icon, XIcon } from 'lucide-react';

import { CloseButton, IconButton } from './ui/Button';
import { ConfirmDialog } from './ui/ConfirmDialog';
import { MenuContent } from './ui/Menu';
import { OpenProjectDialog } from './OpenProjectDialog';
import { RenameDialog } from './ui/RenameDialog';
import { Tabs } from './ui/Tabs';
import { Tooltip } from './ui/Tooltip';
import type { Project } from '../types';
import { exportOpenProject } from '../projects/projectFile';
import { useProjectActions } from '../projects/useProjectActions';
import { useWorkbench } from '../WorkbenchContext';

/**
 * Document-style tabs for the open projects (the session), immediately right
 * of the Invoke control. Tabs are not the library: closing one only removes
 * it from the session — the project stays saved and reopens through the
 * folder button's Open Project dialog. Closing the last tab lands on Home.
 *
 * Each tab is a container with two real buttons (select + close) rather than
 * the prototype's invalid button-nested-in-button, so keyboard focus and
 * click targets behave correctly. Right-clicking a tab opens a context menu
 * with rename, details, export, close, and delete.
 */
export const ProjectTabs = () => {
  const { state, activeProject, dispatch } = useWorkbench();
  const { closeProject, deleteProject } = useProjectActions();
  const [menuTarget, setMenuTarget] = useState<{ project: Project; x: number; y: number } | null>(null);
  const [renameTarget, setRenameTarget] = useState<Project | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null);
  const [isOpenDialogVisible, setIsOpenDialogVisible] = useState(false);

  const onSwitchProject = (projectId: string) => {
    dispatch({ projectId, type: 'switchProject' });
  };

  const openContextMenu = (project: Project, event: MouseEvent) => {
    event.preventDefault();
    setMenuTarget({ project, x: event.clientX, y: event.clientY });
  };

  return (
    <>
      <Tabs.Root minW="max-content" variant="subtle" value={activeProject.id} h="full" w="full">
        <Tabs.List flex="1 1 auto" h="full" py="1">
          {state.projects.map((project) => (
            <Tabs.Trigger
              key={project.id}
              value={project.id}
              onClick={() => onSwitchProject(project.id)}
              onContextMenu={(event) => openContextMenu(project, event)}
              fontSize="xs"
              h="full"
            >
              {project.name}
              <CloseButton
                size="2xs"
                me="-2"
                as="span"
                role="button"
                aria-label={`Close ${project.name}`}
                onClick={(event) => {
                  event.stopPropagation();
                  closeProject(project);
                }}
              />
            </Tabs.Trigger>
          ))}
          <Tooltip content="Create new project" showArrow>
            <IconButton
              aria-label="Create new project"
              flexShrink={0}
              size="xs"
              variant="ghost"
              alignSelf="center"
              ms="2"
              onClick={() => dispatch({ type: 'createProject' })}
            >
              <PlusIcon />
            </IconButton>
          </Tooltip>
          <Tooltip content="Open project" showArrow>
            <IconButton
              aria-label="Open project"
              flexShrink={0}
              size="xs"
              variant="ghost"
              alignSelf="center"
              onClick={() => setIsOpenDialogVisible(true)}
            >
              <FolderOpenIcon />
            </IconButton>
          </Tooltip>
        </Tabs.List>
      </Tabs.Root>
      <ProjectTabContextMenu
        target={menuTarget}
        onClose={() => setMenuTarget(null)}
        onCloseProject={closeProject}
        onDelete={(project) => setDeleteTarget(project)}
        onRename={(project) => setRenameTarget(project)}
      />
      <RenameDialog
        initialName={renameTarget?.name ?? ''}
        isOpen={renameTarget !== null}
        onClose={() => setRenameTarget(null)}
        onSubmit={(name) => {
          if (renameTarget) {
            dispatch({ name, projectId: renameTarget.id, type: 'renameProject' });
          }
        }}
      />
      <ConfirmDialog
        body={`Delete "${deleteTarget?.name ?? ''}"? The project and its saved copy on the server are removed permanently. To keep it in your library, close the tab instead.`}
        confirmLabel="Delete project"
        isOpen={deleteTarget !== null}
        title="Delete project?"
        onClose={() => setDeleteTarget(null)}
        onConfirm={async () => {
          if (deleteTarget) {
            await deleteProject(deleteTarget);
          }
        }}
      />
      <OpenProjectDialog isOpen={isOpenDialogVisible} onClose={() => setIsOpenDialogVisible(false)} />
    </>
  );
};

const ProjectTabContextMenu = ({
  onClose,
  onCloseProject,
  onDelete,
  onRename,
  target,
}: {
  onClose: () => void;
  onCloseProject: (project: Project) => void;
  onDelete: (project: Project) => void;
  onRename: (project: Project) => void;
  target: { project: Project; x: number; y: number } | null;
}) => {
  const { dispatch } = useWorkbench();
  const targetRef = useRef(target);

  targetRef.current = target;

  const openProjectDetails = (project: Project) => {
    dispatch({ projectId: project.id, type: 'switchProject' });

    // Reveal the Project panel wherever the project already shows it; default
    // to enabling it in the right rail. Both actions operate on the active
    // project, which the switch above just made this one.
    const { left, right } = project.widgetRegions;
    const region = right.enabledWidgetIds.includes('project')
      ? 'right'
      : left.enabledWidgetIds.includes('project')
        ? 'left'
        : null;

    if (region) {
      dispatch({ region, type: 'selectRegionWidget', widgetId: 'project' });
    } else {
      dispatch({ region: 'right', type: 'toggleRegionWidget', widgetId: 'project' });
    }
  };

  return (
    <Menu.Root
      key={target?.project.id ?? 'closed'}
      lazyMount
      open={target !== null}
      positioning={{
        getAnchorRect: () => {
          const currentTarget = targetRef.current;

          return currentTarget ? { height: 1, width: 1, x: currentTarget.x, y: currentTarget.y } : null;
        },
        placement: 'bottom-start',
      }}
      unmountOnExit
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Menu.Positioner>
          {target ? (
            <MenuContent minW="44">
              <Menu.Item value="rename" onClick={() => onRename(target.project)}>
                <Icon as={PencilIcon} boxSize="3.5" />
                Rename…
              </Menu.Item>
              <Menu.Item value="details" onClick={() => openProjectDetails(target.project)}>
                <Icon as={FolderCogIcon} boxSize="3.5" />
                Project details
              </Menu.Item>
              <Menu.Item value="export" onClick={() => exportOpenProject(target.project)}>
                <Icon as={FileDownIcon} boxSize="3.5" />
                Export
              </Menu.Item>
              <Menu.Separator />
              <Menu.Item value="close" onClick={() => onCloseProject(target.project)}>
                <Icon as={XIcon} boxSize="3.5" />
                Close
              </Menu.Item>
              <Menu.Separator />
              <Menu.Item
                color="fg.error"
                value="delete"
                _hover={{ bg: 'bg.error', color: 'fg.error' }}
                onClick={() => onDelete(target.project)}
              >
                <Icon as={Trash2Icon} boxSize="3.5" />
                Delete project…
              </Menu.Item>
            </MenuContent>
          ) : null}
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
