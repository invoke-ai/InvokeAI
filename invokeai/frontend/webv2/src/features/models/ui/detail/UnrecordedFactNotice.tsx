import type { UnrecordedModelFact } from '@features/models/core/taxonomy';

import { HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { RefreshCcwIcon, TriangleAlertIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const MESSAGE_KEYS: Record<UnrecordedModelFact, string> = {
  adapter_type: 'models.adapterTypeUnknown',
  native_scale: 'models.nativeScaleUnknown',
};

/**
 * Explains what a model installed before the probe recorded it is missing, and offers Re-identify. `onReidentify` is
 * null for users who cannot manage models; they are told to ask an administrator instead.
 */
export const UnrecordedFactNotice = ({
  fact,
  onReidentify,
}: {
  fact: UnrecordedModelFact;
  onReidentify: (() => void) | null;
}) => {
  const { t } = useTranslation();

  return (
    <HStack
      align="start"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderRadius="md"
      borderWidth="1px"
      gap="2"
      p="2.5"
      wrap="wrap"
    >
      <Icon as={TriangleAlertIcon} boxSize="3.5" color="fg.warning" flexShrink={0} mt="0.5" />
      <Stack flex="1 1 14rem" gap="0.5" minW="0">
        <Text color="fg.muted" fontSize="xs">
          {t(MESSAGE_KEYS[fact])}
        </Text>
        {onReidentify ? null : (
          <Text color="fg.subtle" fontSize="xs">
            {t('models.reidentifyAskAdmin')}
          </Text>
        )}
      </Stack>
      {onReidentify ? (
        <Button flexShrink={0} ms="auto" variant="outline" onClick={onReidentify}>
          <Icon as={RefreshCcwIcon} boxSize="3" />
          {t('models.reidentify')}
        </Button>
      ) : null}
    </HStack>
  );
};
