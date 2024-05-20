import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { PiCopyBold } from 'react-icons/pi';

function handleCopy(sessionId: string) {
  navigator.clipboard.writeText(sessionId);
}

export default function ToastDescription({ message, sessionId }: { message: string; sessionId: string }) {
  return (
    <Flex flexDir="column">
      <Text fontSize="md">{message}</Text>
      <Flex gap="2" alignItems="center">
        <Text fontSize="sm">
          {t('toast.sessionReference')}: {sessionId}
        </Text>
        <IconButton
          size="sm"
          aria-label="Copy"
          colorScheme="error"
          icon={<PiCopyBold />}
          onClick={handleCopy.bind(null, sessionId)}
        />
      </Flex>
    </Flex>
  );
}
