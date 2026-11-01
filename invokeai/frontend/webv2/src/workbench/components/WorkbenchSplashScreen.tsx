import splashImageUrl from '@assets/SplashImage.webp';
import { Badge, Box, Flex, Heading, Spinner, Stack, Text, VStack } from '@chakra-ui/react';
import { APP_VERSION } from '@platform/runtime/appMetadata';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

const splashImageStyle = { backgroundPosition: 'center', backgroundSize: 'cover' } as const;
// Side by side where the card fits (it needs about 680px), stacked below that with the artwork as a banner.
const CARD_DIRECTION = { base: 'column', md: 'row' } as const;
const ART_PADDING = { base: '6', md: '14' } as const;
const ART_ASPECT = { base: 'auto', md: '1/1' } as const;
const MARK_SIZE = 164;
const TEXT_MIN_WIDTH = { base: '0', md: '80' } as const;

export const WorkbenchSplashScreen = ({
  action,
  messageKey = 'splash.loadingWorkspace',
}: {
  /** Shown under the status, such as a way out of a launch that is opening the last project. */
  action?: ReactNode;
  messageKey?: string;
}) => {
  const { t } = useTranslation();
  const [year] = useState(() => new Date().getFullYear());

  return (
    <Flex align="center" aria-busy="true" justify="center" layerStyle="dotGrid" minH="100dvh" p="4" w="full">
      <Box
        position="relative"
        bg="bg"
        color="fg"
        borderColor="border.subtle"
        borderRadius="3xl"
        borderWidth="1px"
        maxW="2xl"
        p="6"
        shadow="sm"
        w="full"
      >
        <Stack align="stretch" direction={CARD_DIRECTION} gap="8">
          <Flex
            align="center"
            aspectRatio={ART_ASPECT}
            bgImage={`url(${splashImageUrl})`}
            flexShrink={0}
            justify="center"
            p={ART_PADDING}
            rounded="xl"
            style={splashImageStyle}
          >
            <InvokeMark size={MARK_SIZE} />
          </Flex>
          <VStack align="start" flex="1" gap="3" minH="full" minW={TEXT_MIN_WIDTH} textAlign="start" py="2">
            <VStack align="start" gap="1.5">
              <Heading size="4xl">{t('app.name')}</Heading>
              <Badge fontFamily="mono" size="lg" variant="subtle">
                v{APP_VERSION}
              </Badge>
            </VStack>
            <Text fontSize="xl">{t('splash.tagline')}</Text>
            <Text fontSize="md">{t('splash.artworkBy', { artist: 'Jonathan Pollack' })}</Text>
            <Flex alignItems="center" gap="2" mt="auto">
              <Spinner />
              {/* Only the changing status is announced, not the tagline, credit or copyright. */}
              <Text fontSize="md" flex="1" role="status">
                {t(messageKey)}
              </Text>
            </Flex>
            {action}
            <Text fontSize="xs" fontFamily="mono">
              {t('splash.copyright', { year })}
            </Text>
          </VStack>
        </Stack>
      </Box>
    </Flex>
  );
};
