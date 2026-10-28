import { HStack, Icon, Text } from '@chakra-ui/react';
import { TriangleAlertIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Sits beside a retained draft's save actions. */
export const UnsavedChangesLabel = () => {
  const { t } = useTranslation();

  return (
    <Text color="fg.muted" flexShrink={0} fontSize="xs">
      {t('models.unsavedChanges')}
    </Text>
  );
};

/** The server changed a field the user also edited; the form keeps the user's value until reset or saved. */
export const DraftConflictNotice = () => {
  const { t } = useTranslation();

  return (
    <HStack align="start" gap="1.5" role="status">
      <Icon as={TriangleAlertIcon} boxSize="3" color="fg.warning" flexShrink={0} mt="0.5" />
      <Text color="fg.muted" fontSize="xs">
        {t('models.changedElsewhere')}
      </Text>
    </HStack>
  );
};
