import { Box, Flex, HStack, Icon, IconButton } from '@chakra-ui/react';
import { PiPlusBold, PiXBold } from 'react-icons/pi';

import { useWorkbench } from '../WorkbenchContext';
import type { Project } from '../types';

/**
 * Document-style project tabs, immediately right of the Invoke control.
 *
 * Each tab is a container with two real buttons (select + close) rather than the
 * prototype's invalid button-nested-in-button, so keyboard focus and click
 * targets behave correctly.
 */
export const ProjectTabs = () => {
  const { state, activeProject, dispatch } = useWorkbench();

  return (
    <HStack flex="1" gap="1" h="full" minW="0" overflowX="auto">
      {state.projects.map((project) => (
        <ProjectTab
          key={project.id}
          project={project}
          isActive={project.id === activeProject.id}
          canClose={state.projects.length > 1}
          onSelect={() => dispatch({ projectId: project.id, type: 'switchProject' })}
          onClose={() => dispatch({ projectId: project.id, type: 'closeProject' })}
        />
      ))}
      <IconButton
        aria-label="New project"
        color="fg.muted"
        flexShrink={0}
        size="xs"
        variant="ghost"
        _hover={{ bg: 'bg.surface', color: 'fg.default' }}
        onClick={() => dispatch({ type: 'createProject' })}
      >
        <PiPlusBold />
      </IconButton>
    </HStack>
  );
};

interface ProjectTabProps {
  project: Project;
  isActive: boolean;
  canClose: boolean;
  onSelect: () => void;
  onClose: () => void;
}

const ProjectTab = ({ project, isActive, canClose, onSelect, onClose }: ProjectTabProps) => (
  <Flex
    align="center"
    bg={isActive ? 'bg.panel' : 'transparent'}
    borderWidth="1px"
    borderColor={isActive ? 'border.emphasis' : 'transparent'}
    color={isActive ? 'fg.default' : 'fg.muted'}
    flexShrink={0}
    gap="2"
    h="8"
    maxW="11rem"
    minW="7.5rem"
    px="2"
    rounded="md"
    transition="background 0.12s ease, color 0.12s ease"
    _hover={{ color: 'fg.default' }}
  >
    <Box
      aria-current={isActive ? 'page' : undefined}
      as="button"
      flex="1"
      fontSize="xs"
      fontWeight={isActive ? '600' : '500'}
      minW="0"
      textAlign="left"
      truncate
      onClick={onSelect}
    >
      {project.name}
    </Box>
    {canClose ? (
      <IconButton
        aria-label={`Close ${project.name}`}
        color="fg.subtle"
        size="2xs"
        variant="ghost"
        _hover={{ bg: 'bg.surfaceRaised', color: 'fg.default' }}
        onClick={onClose}
      >
        <Icon as={PiXBold} boxSize="3" />
      </IconButton>
    ) : null}
  </Flex>
);
