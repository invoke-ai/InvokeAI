import { CloseButton, IconButton } from './ui/Button';
import { Tabs } from './ui/Tabs';
import { PlusIcon } from 'lucide-react';

import { Tooltip } from './ui/Tooltip';
import { useWorkbench } from '../WorkbenchContext';

/**
 * Document-style project tabs, immediately right of the Invoke control.
 *
 * Each tab is a container with two real buttons (select + close) rather than the
 * prototype's invalid button-nested-in-button, so keyboard focus and click
 * targets behave correctly.
 */
export const ProjectTabs = () => {
  const { state, activeProject, dispatch } = useWorkbench();

  const onSwitchProject = (projectId: string) => {
    dispatch({ projectId, type: 'switchProject' });
  };

  return (
    <Tabs.Root minW="max-content" variant="subtle" value={activeProject.id} h="full" w="full">
      <Tabs.List flex="1 1 auto" h="full" py="1">
        {state.projects.map((project) => (
          <Tabs.Trigger
            key={project.id}
            value={project.id}
            onClick={() => onSwitchProject(project.id)}
            fontSize="xs"
            h="full"
          >
            {project.name}
            {/*{project.id !== activeProject.id && project.queue.items.some((item) => item.resultImages?.length) ? (
                  <Badge colorPalette="blue" size="xs">
                    Results
                  </Badge>
                ) : null}*/}
            {state.projects.length > 1 && (
              <CloseButton
                size="2xs"
                me="-2"
                as="span"
                role="button"
                onClick={(event) => {
                  event.stopPropagation();
                  dispatch({ projectId: project.id, type: 'closeProject' });
                }}
              />
            )}
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
      </Tabs.List>
    </Tabs.Root>
  );
};
