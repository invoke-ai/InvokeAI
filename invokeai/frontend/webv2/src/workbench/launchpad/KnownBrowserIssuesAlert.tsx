import { Alert, Code, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import {
  detectKnownBrowserIssues,
  type KnownBrowserIssue,
  type KnownBrowserIssueWorkaround,
} from '@platform/browser/knownBrowserIssues';
import { useMountEffect } from '@platform/react/useMountEffect';
import { IconButton, toaster } from '@platform/ui';
import { CopyIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

export const KnownBrowserIssuesAlert = () => {
  const [issues, setIssues] = useState<readonly KnownBrowserIssue[]>([]);

  useMountEffect(() => {
    let isMounted = true;

    void detectKnownBrowserIssues().then((detectedIssues) => {
      if (isMounted) {
        setIssues(detectedIssues);
      }
    });

    return () => {
      isMounted = false;
    };
  });

  if (issues.length === 0) {
    return null;
  }

  return (
    <Stack gap="2">
      {issues.map((issue) => (
        <BrowserIssueAlert issue={issue} key={issue.id} />
      ))}
    </Stack>
  );
};

const BrowserIssueAlert = ({ issue }: { issue: KnownBrowserIssue }) => {
  const { t } = useTranslation();

  return (
    <Alert.Root borderRadius="md" role="alert" size="sm" status={issue.severity} variant="surface">
      <Alert.Indicator alignSelf="start" mt="0.5" />
      <Alert.Content gap="2" minW="0">
        <Stack gap="0.5">
          <Alert.Title fontSize="sm">{t(issue.titleKey)}</Alert.Title>
          <Alert.Description fontSize="xs">{t(issue.descriptionKey)}</Alert.Description>
        </Stack>
        <Stack gap="1.5">
          {issue.workarounds.map((workaround) => (
            <BrowserIssueWorkaround key={workaround.id} workaround={workaround} />
          ))}
        </Stack>
      </Alert.Content>
    </Alert.Root>
  );
};

const BrowserIssueWorkaround = ({ workaround }: { workaround: KnownBrowserIssueWorkaround }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="1">
      <Text color="fg.muted" fontSize="xs" overflowWrap="anywhere">
        {t(workaround.instructionKey)}
      </Text>
      {workaround.copyableValue ? <CopyableBrowserSetting value={workaround.copyableValue} /> : null}
    </Stack>
  );
};

const CopyableBrowserSetting = ({ value }: { value: string }) => {
  const { t } = useTranslation();
  const copySetting = useCallback(async () => {
    try {
      if (!navigator.clipboard) {
        throw new Error('Clipboard API unavailable');
      }

      await navigator.clipboard.writeText(value);
      toaster.create({
        title: t('launchpad.browserIssues.settingCopied'),
        type: 'success',
      });
    } catch {
      toaster.create({
        title: t('launchpad.browserIssues.copyFailed'),
        type: 'error',
      });
    }
  }, [t, value]);
  const handleCopy = useCallback(() => void copySetting(), [copySetting]);

  return (
    <HStack
      alignSelf="start"
      bg="bg.inset"
      borderColor="border.muted"
      borderRadius="sm"
      borderWidth="1px"
      gap="1"
      maxW="full"
      minW="0"
      p="1"
    >
      <Code bg="transparent" fontSize="2xs" minW="0" overflowWrap="anywhere" px="1" whiteSpace="normal">
        {value}
      </Code>
      <IconButton
        aria-label={t('launchpad.browserIssues.copySetting')}
        flexShrink={0}
        size="2xs"
        title={t('launchpad.browserIssues.copySetting')}
        variant="ghost"
        onClick={handleCopy}
      >
        <Icon as={CopyIcon} boxSize="3" />
      </IconButton>
    </HStack>
  );
};
