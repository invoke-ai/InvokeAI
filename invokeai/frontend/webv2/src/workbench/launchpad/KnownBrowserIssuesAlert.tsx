import { Alert, Code, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { canCopyText, copyText } from '@platform/browser/clipboard';
import {
  detectKnownBrowserIssues,
  type DetectedBrowserIssue,
  type KnownBrowserIssueId,
  type KnownBrowserIssueWorkaround,
} from '@platform/browser/knownBrowserIssues';
import { IconButton } from '@platform/ui/Button';
import { toaster } from '@platform/ui/toaster';
import { Tooltip } from '@platform/ui/Tooltip';
import { CopyIcon, XIcon } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';

/** Per origin by nature of `localStorage`, so another address the viewer opens shows the notice again. */
const DISMISSED_STORAGE_KEY_PREFIX = 'invokeai:v7:webv2:browser-issue-dismissed:';

const isRememberedDismissed = (id: KnownBrowserIssueId): boolean => {
  try {
    return window.localStorage.getItem(`${DISMISSED_STORAGE_KEY_PREFIX}${id}`) === 'true';
  } catch {
    return false;
  }
};

const rememberDismissed = (id: KnownBrowserIssueId): void => {
  try {
    window.localStorage.setItem(`${DISMISSED_STORAGE_KEY_PREFIX}${id}`, 'true');
  } catch {
    // Storage unavailable: the notice is hidden for this page and shows again on the next visit.
  }
};

/** The page heading above the alerts, where focus goes once the last one is dismissed. */
const findPrecedingHeading = (element: HTMLElement): HTMLElement | null =>
  [...document.querySelectorAll<HTMLElement>('h1, h2')]
    .filter((heading) => heading.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_FOLLOWING)
    .at(-1) ?? null;

export const KnownBrowserIssuesAlert = () => {
  // The probe is cheap and its result is fixed for the page's lifetime, so it runs once per mount.
  const [issues] = useState(() => detectKnownBrowserIssues());
  const [dismissedIds, setDismissedIds] = useState<ReadonlySet<KnownBrowserIssueId>>(
    () => new Set(issues.filter((issue) => issue.dismissible && isRememberedDismissed(issue.id)).map(({ id }) => id))
  );
  const stackRef = useRef<HTMLDivElement | null>(null);
  const dismiss = useCallback((id: KnownBrowserIssueId) => {
    rememberDismissed(id);
    const stack = stackRef.current;
    const alerts = stack ? [...stack.querySelectorAll<HTMLElement>('[data-browser-issue]')] : [];
    const index = alerts.findIndex((alert) => alert.dataset.browserIssue === id);
    // Focus never falls to the body: the next alert, else the previous one, else the page heading.
    const nextFocus = alerts[index + 1] ?? alerts[index - 1] ?? (stack ? findPrecedingHeading(stack) : null);

    flushSync(() => setDismissedIds((previous) => new Set(previous).add(id)));
    if (nextFocus) {
      if (!nextFocus.hasAttribute('tabindex')) {
        nextFocus.tabIndex = -1;
      }
      nextFocus.focus();
    }
  }, []);
  const visible = issues.filter((issue) => !dismissedIds.has(issue.id));

  if (visible.length === 0) {
    return null;
  }

  return (
    <Stack ref={stackRef} gap="2">
      {visible.map((issue) => (
        <BrowserIssueAlert issue={issue} key={issue.id} onDismiss={issue.dismissible ? dismiss : undefined} />
      ))}
    </Stack>
  );
};

const BrowserIssueAlert = ({
  issue,
  onDismiss,
}: {
  issue: DetectedBrowserIssue;
  onDismiss?: (id: KnownBrowserIssueId) => void;
}) => {
  const { t } = useTranslation();
  const handleDismiss = useCallback(() => onDismiss?.(issue.id), [issue.id, onDismiss]);

  return (
    <Alert.Root
      borderRadius="md"
      data-browser-issue={issue.id}
      role="alert"
      status={issue.severity}
      tabIndex={-1}
      variant="surface"
    >
      <Alert.Indicator alignSelf="start" mt="0.5" />
      <Alert.Content gap="2" minW="0">
        <Stack gap="0.5">
          <Alert.Title fontSize="lg">{t(issue.titleKey)}</Alert.Title>
          <Alert.Description fontSize="md">{t(issue.descriptionKey)}</Alert.Description>
        </Stack>
        <Stack gap="1.5">
          {issue.workarounds.map((workaround) => (
            <BrowserIssueWorkaround key={workaround.id} workaround={workaround} />
          ))}
        </Stack>
      </Alert.Content>
      {onDismiss ? (
        <Tooltip content={t('launchpad.browserIssues.dismiss')}>
          <IconButton
            alignSelf="start"
            aria-label={t('launchpad.browserIssues.dismiss')}
            flexShrink={0}
            size="sm"
            variant="ghost"
            onClick={handleDismiss}
          >
            <Icon as={XIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
    </Alert.Root>
  );
};

const BrowserIssueWorkaround = ({ workaround }: { workaround: KnownBrowserIssueWorkaround }) => {
  const { t } = useTranslation();

  return (
    <Stack gap="1">
      <Text color="fg.muted" fontSize="md" overflowWrap="anywhere">
        {t(workaround.instructionKey)}
      </Text>
      {workaround.copyableValue ? <CopyableBrowserSetting value={workaround.copyableValue} /> : null}
    </Stack>
  );
};

const CopyableBrowserSetting = ({ value }: { value: string }) => {
  const { t } = useTranslation();
  const canCopy = canCopyText();
  const copySetting = useCallback(async () => {
    try {
      await copyText(value);
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
      <Code bg="transparent" fontSize="xs" minW="0" overflowWrap="anywhere" px="1" whiteSpace="normal">
        {value}
      </Code>
      <IconButton
        tooltip={t('launchpad.browserIssues.copySetting')}
        unavailableReason={canCopy ? undefined : t('common.copyUnavailable')}
        aria-label={t('launchpad.browserIssues.copySetting')}
        flexShrink={0}
        size="sm"
        variant="ghost"
        onClick={handleCopy}
      >
        <Icon as={CopyIcon} boxSize="3" />
      </IconButton>
    </HStack>
  );
};
