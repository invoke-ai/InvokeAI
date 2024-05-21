import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { PiCopyBold } from 'react-icons/pi';

function onCopy(sessionId: string) {
  navigator.clipboard.writeText(sessionId);
}

type Props = { message: string; sessionId: string };

export default function ToastWithSessionRefDescription({ message, sessionId }: Props) {
  return (
    <Flex flexDir="column">
      <Text fontSize="md">{message}</Text>
      <Flex gap="2" alignItems="center">
        <Text fontSize="sm">{t('toast.sessionRef', { sessionId })}</Text>
        <IconButton
          size="sm"
          aria-label="Copy"
          icon={<PiCopyBold />}
          onClick={onCopy.bind(null, sessionId)}
          variant="ghost"
        />
      </Flex>
    </Flex>
  );
}
