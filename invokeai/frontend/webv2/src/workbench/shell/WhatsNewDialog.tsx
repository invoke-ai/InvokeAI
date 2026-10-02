import { Dialog, HStack, Link, List, Portal, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { DOCS_URL, getReleaseNotesUrl } from '@platform/runtime/appMetadata';
import { CloseButton } from '@platform/ui';
import { InvokeMark } from '@platform/ui/InvokeMark';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { useWhatsNew } from './useWhatsNew';
import { dismissWhatsNew } from './whatsNewStore';

const EMPHASIS = /<StrongComponent>(.*?)<\/StrongComponent>/;

const readItems = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : [];

/** Items may emphasize a phrase with `<StrongComponent>…</StrongComponent>`; split captures land on odd indexes. */
const WhatsNewItem = ({ text }: { text: string }) =>
  text.split(EMPHASIS).map((part, index) =>
    index % 2 === 1 ? (
      <Text key={index} as="span" color="fg" fontWeight="600">
        {part}
      </Text>
    ) : (
      part
    )
  );

/** Mounted only while the notes are open: workbench hotkeys stay quiet under them, as under the other dialogs. */
const WhatsNewModalLayer = () => {
  useMountEffect(() => registerHotkeyModalLayer('whats-new'));

  return null;
};

export const WhatsNewDialog = () => {
  const { t } = useTranslation();
  const { isOpen, isUnseen, version } = useWhatsNew();
  const items = readItems(t('whatsNew.items', { returnObjects: true }));
  const contentRef = useRef<HTMLDivElement | null>(null);
  // Start on the panel, not the first link: focus that opens without a click (the automatic showing) counts as
  // keyboard focus and would ring the link. Tab still reaches the links, and Escape still closes.
  const getInitialFocusEl = useCallback(() => contentRef.current, []);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (event.open) {
        return;
      }

      dismissWhatsNew();

      // Unseen implies loaded preferences; patching before they load would persist defaults over saved choices.
      if (isUnseen) {
        void patchWorkbenchPreferences({ whatsNewSeenVersion: version });
      }
    },
    [isUnseen, version]
  );

  return (
    <Dialog.Root
      initialFocusEl={getInitialFocusEl}
      open={isOpen}
      placement="center"
      size="xs"
      onOpenChange={handleOpenChange}
    >
      {isOpen ? <WhatsNewModalLayer /> : null}
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content ref={contentRef}>
            <Dialog.Header>
              <HStack gap="2.5">
                <InvokeMark size={20} />
                <Dialog.Title fontSize="sm">{t('whatsNew.whatsNewInInvoke')}</Dialog.Title>
                {version ? (
                  <Text color="fg.subtle" fontSize="xs">
                    v{version}
                  </Text>
                ) : null}
              </HStack>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="4">
                <List.Root fontSize="xs" gap="1" ps="4">
                  {items.map((item, index) => (
                    // The catalog order is fixed for a build, so the index is a stable key.
                    <List.Item key={index}>
                      <WhatsNewItem text={item} />
                    </List.Item>
                  ))}
                </List.Root>
                <Stack gap="1">
                  <Link
                    color="accent.fg"
                    fontSize="xs"
                    fontWeight="600"
                    href={getReleaseNotesUrl(version)}
                    rel="noreferrer"
                    target="_blank"
                  >
                    {t('whatsNew.readReleaseNotes')}
                  </Link>
                  <Link
                    color="accent.fg"
                    fontSize="xs"
                    fontWeight="600"
                    href={DOCS_URL}
                    rel="noreferrer"
                    target="_blank"
                  >
                    {t('whatsNew.readTheDocs')}
                  </Link>
                </Stack>
              </Stack>
            </Dialog.Body>
            <Dialog.CloseTrigger asChild>
              <CloseButton />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
