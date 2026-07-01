/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Flex, Icon, Text } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { PlusIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** The grid's leading cell: start a fresh draft in the editor. */
export const NewProjectCard = () => {
  const { t } = useTranslation();

  return (
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
      transition="border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease"
      _hover={{ bg: 'bg.subtle', borderColor: 'border.emphasized' }}
    >
      <Link
        aria-label={t('projects.createNewProject')}
        search={{ new: true }}
        style={{ inset: 0, position: 'absolute' }}
        to="/app"
      />
      <Icon as={PlusIcon} boxSize="6" color="fg.muted" />
      <Text color="fg.muted" fontSize="xs" fontWeight="600">
        {t('projects.newProject')}
      </Text>
    </Flex>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
