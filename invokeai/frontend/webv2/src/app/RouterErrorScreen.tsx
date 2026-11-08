import { Box, Flex, Heading, HStack, Stack, Text } from '@chakra-ui/react';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import { Button } from '@platform/ui/Button';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { useCallback, useId, useState } from 'react';
import { useTranslation } from 'react-i18next';

const COPY_STATUS_MS = 2000;

const reload = () => window.location.reload();

/** What a bug report needs: the error's name, message and stack, plus where the app was. */
const formatErrorDetails = (error: unknown, location: string): string => {
  const summary = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  // V8 stacks begin with the summary; others (Firefox, Safari) are frames only.
  const stack = error instanceof Error ? error.stack : undefined;
  const body = stack ? (stack.startsWith(summary) ? stack : `${summary}\n${stack}`) : summary;

  return `${body}\nLocation: ${location}`;
};

/**
 * The screen a route that failed to render or load leaves behind: ways back (reload, Home) and the details for a
 * report. It sits in the router's eager chunk, so it reads only semantic tokens and small platform controls.
 */
export const RouterErrorScreen = ({ error, onGoHome }: { error: unknown; onGoHome: () => void }) => {
  const { t } = useTranslation();
  const detailsId = useId();
  const [isShowingDetails, setIsShowingDetails] = useState(false);
  const [copyStatus, setCopyStatus] = useState<'copied' | 'failed' | null>(null);
  const details = formatErrorDetails(error, window.location.href);

  const copyDetails = useCallback(() => {
    void copyText(details).then(
      () => setCopyStatus('copied'),
      () => {
        // A failed copy shows the details instead, so they can be selected and copied by hand.
        setCopyStatus('failed');
        setIsShowingDetails(true);
      }
    );
    window.setTimeout(() => setCopyStatus(null), COPY_STATUS_MS);
  }, [details]);
  const toggleDetails = useCallback(() => setIsShowingDetails((isShowing) => !isShowing), []);

  return (
    <Flex align="center" bg="bg" color="fg" justify="center" minH="100dvh" p="6">
      <Stack gap="6" maxW="xl" w="full">
        <Stack align="center" gap="4">
          <InvokeMark />
          <Stack align="center" gap="1">
            <Heading as="h1" fontSize="2xl" fontWeight="700">
              {t('errorScreen.title')}
            </Heading>
            <Text color="fg.muted" fontSize="lg" textAlign="center" textWrap="pretty">
              {t('errorScreen.description')}
            </Text>
          </Stack>
        </Stack>
        <HStack flexWrap="wrap" gap="2" justify="center">
          <Button size="lg" onClick={reload}>
            {t('common.reload')}
          </Button>
          <Button size="lg" variant="outline" onClick={onGoHome}>
            {t('errorScreen.goHome')}
          </Button>
          {canCopyText() ? (
            <Button size="lg" variant="outline" onClick={copyDetails}>
              {t('errorScreen.copyDetails')}
            </Button>
          ) : null}
          <Button
            aria-controls={isShowingDetails ? detailsId : undefined}
            aria-expanded={isShowingDetails}
            size="lg"
            variant="ghost"
            onClick={toggleDetails}
          >
            {isShowingDetails ? t('errorScreen.hideDetails') : t('errorScreen.showDetails')}
          </Button>
        </HStack>
        <Text
          aria-live="polite"
          color={copyStatus === 'failed' ? 'fg.error' : 'fg.muted'}
          fontSize="md"
          minH="5"
          textAlign="center"
        >
          {copyStatus === 'copied'
            ? t('errorScreen.copied')
            : copyStatus === 'failed'
              ? t('errorScreen.copyFailed')
              : null}
        </Text>
        {isShowingDetails ? (
          <Box
            as="pre"
            bg="bg.subtle"
            borderColor="border.subtle"
            borderWidth="1px"
            fontFamily="mono"
            fontSize="sm"
            id={detailsId}
            maxH="50dvh"
            overflow="auto"
            p="3"
            rounded="md"
            whiteSpace="pre-wrap"
            wordBreak="break-word"
          >
            {details}
          </Box>
        ) : null}
      </Stack>
    </Flex>
  );
};
