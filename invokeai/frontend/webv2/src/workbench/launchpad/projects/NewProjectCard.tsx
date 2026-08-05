import { Flex, Icon, Text } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { PlusIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** The library's leading cell: start a fresh draft in the editor. */

const NEW_PROJECT_SEARCH = { new: true } as const;
const LINK_STYLE = { inset: 0, position: 'absolute' } as const;
const CARD_HOVER = { bg: 'bg.subtle', borderColor: 'border.emphasized' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';

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
      minH="28"
      position="relative"
      rounded="lg"
      transition={CARD_TRANSITION}
      _hover={CARD_HOVER}
    >
      <Link aria-label={t('projects.createNewProject')} search={NEW_PROJECT_SEARCH} style={LINK_STYLE} to="/app" />
      <Icon as={PlusIcon} boxSize="5" color="fg.muted" />
      <Text color="fg.muted" fontSize="xs" fontWeight="600">
        {t('projects.newProject')}
      </Text>
    </Flex>
  );
};
