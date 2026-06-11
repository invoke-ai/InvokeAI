import { Flex, Icon, Text } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { PlusIcon } from 'lucide-react';

/** The grid's leading cell: start a fresh draft in the editor. */
export const NewProjectCard = () => (
  <Flex
    align="center"
    borderColor="border.subtle"
    borderStyle="dashed"
    borderWidth="1.5px"
    direction="column"
    gap="2"
    justify="center"
    minH="40"
    position="relative"
    rounded="lg"
    transition="border-color 0.15s ease, background 0.15s ease"
    _hover={{ bg: 'bg.surface', borderColor: 'border.emphasized' }}
  >
    <Link aria-label="Create new project" search={{ new: true }} style={{ inset: 0, position: 'absolute' }} to="/app" />
    <Icon as={PlusIcon} boxSize="6" color="fg.muted" />
    <Text color="fg.muted" fontSize="xs" fontWeight="600">
      New project
    </Text>
  </Flex>
);
