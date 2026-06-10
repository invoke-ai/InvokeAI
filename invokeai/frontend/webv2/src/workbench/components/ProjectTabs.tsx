import { Badge, CloseButton, IconButton, ScrollArea, Tabs } from '@chakra-ui/react';
import { PiPlusBold } from 'react-icons/pi';

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
    <ScrollArea.Root flex="1" minW="0" size="xs" variant="hover">
      <ScrollArea.Viewport h="full" w="full">
        <Tabs.Root minW="max-content" variant="outline" value={activeProject.id}>
          <Tabs.List flex="1 1 auto" h="full">
            {state.projects.map((project) => (
              <Tabs.Trigger
                key={project.id}
                value={project.id}
                onClick={() => onSwitchProject(project.id)}
                fontSize="xs"
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
            <Tooltip content="Create new project">
              <IconButton
                aria-label="Create new project"
                color="fg.muted"
                flexShrink={0}
                size="xs"
                variant="ghost"
                alignSelf="center"
                ms="2"
                onClick={() => dispatch({ type: 'createProject' })}
              >
                <PiPlusBold />
              </IconButton>
            </Tooltip>
          </Tabs.List>
        </Tabs.Root>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar orientation="horizontal">
        <ScrollArea.Thumb />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  );
};
