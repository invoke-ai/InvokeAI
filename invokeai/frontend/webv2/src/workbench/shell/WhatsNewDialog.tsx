import type { LucideIcon } from 'lucide-react';

import v7LogoUrl from '@assets/V7Logo.webp';
import { Badge, Box, Dialog, Flex, Grid, Image, Link, List, Portal, Stack, Text, VStack } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { DOCS_URL, getReleaseNotesUrl } from '@platform/runtime/appMetadata';
import { Button, CloseButton } from '@platform/ui';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { BookOpenIcon, BoxesIcon, BrushIcon, FolderIcon, PanelsTopLeftIcon, ScrollTextIcon } from 'lucide-react';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { useWhatsNew } from './useWhatsNew';
import { dismissWhatsNew } from './whatsNewStore';

/** Headline features get an icon and a title; the `whatsNew.items` catalog lists the smaller notes beneath them. */
const HIGHLIGHTS: readonly { descriptionKey: string; icon: LucideIcon; titleKey: string }[] = [
  {
    descriptionKey: 'whatsNew.highlights.interface.description',
    icon: PanelsTopLeftIcon,
    titleKey: 'whatsNew.highlights.interface.title',
  },
  {
    descriptionKey: 'whatsNew.highlights.projects.description',
    icon: FolderIcon,
    titleKey: 'whatsNew.highlights.projects.title',
  },
  {
    descriptionKey: 'whatsNew.highlights.canvas.description',
    icon: BrushIcon,
    titleKey: 'whatsNew.highlights.canvas.title',
  },
  {
    descriptionKey: 'whatsNew.highlights.models.description',
    icon: BoxesIcon,
    titleKey: 'whatsNew.highlights.models.title',
  },
];

const readItems = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : [];

/** Mounted only while the notes are open: workbench hotkeys stay quiet under them, as under the other dialogs. */
const WhatsNewModalLayer = () => {
  useMountEffect(() => registerHotkeyModalLayer('whats-new'));

  return null;
};

/**
 * The modal traps focus, so focus outside it is never the user leaving. Under StrictMode the first, lazily loaded
 * mount restores focus to the launcher button mid-open, and the stale dismiss layer closed the notes on that.
 */
const keepOpenOnFocusOutside = (event: { preventDefault: () => void }) => event.preventDefault();

export const WhatsNewDialog = ({ onExitComplete }: { onExitComplete?: () => void }) => {
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
      scrollBehavior="inside"
      size="lg"
      onExitComplete={onExitComplete}
      onFocusOutside={keepOpenOnFocusOutside}
      onOpenChange={handleOpenChange}
    >
      {isOpen ? <WhatsNewModalLayer /> : null}
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content ref={contentRef} overflow="hidden">
            <Dialog.Header justifyContent="center" pb="4" pt="8">
              <VStack gap="3">
                <Image alt="" boxSize="24" draggable={false} src={v7LogoUrl} />
                <VStack gap="1.5">
                  <Dialog.Title textStyle="3xl">{t('whatsNew.whatsNewInInvoke')}</Dialog.Title>
                  {version ? (
                    <Badge fontFamily="mono" size="lg" variant="subtle">
                      v{version}
                    </Badge>
                  ) : null}
                </VStack>
              </VStack>
            </Dialog.Header>
            <Dialog.Body px="6">
              <Stack gap="5">
                <Grid gap="3" templateColumns="repeat(auto-fit, minmax(15rem, 1fr))">
                  {HIGHLIGHTS.map(({ descriptionKey, icon: HighlightIcon, titleKey }) => (
                    <Flex
                      key={titleKey}
                      align="start"
                      bg="bg.muted"
                      borderColor="border.subtle"
                      borderRadius="lg"
                      borderWidth="1px"
                      gap="3"
                      p="3"
                    >
                      <Flex
                        align="center"
                        bg="accent.subtle"
                        borderRadius="md"
                        boxSize="8"
                        color="accent.fg"
                        flexShrink="0"
                        justify="center"
                      >
                        <HighlightIcon aria-hidden size={16} />
                      </Flex>
                      <Box minW="0">
                        <Text fontWeight="600" textStyle="lg">
                          {t(titleKey)}
                        </Text>
                        <Text color="fg.muted" textStyle="md">
                          {t(descriptionKey)}
                        </Text>
                      </Box>
                    </Flex>
                  ))}
                </Grid>
                {items.length > 0 ? (
                  <Stack gap="1.5">
                    <Text color="fg.subtle" fontWeight="600" textStyle="xs" textTransform="uppercase">
                      {t('whatsNew.alsoNew')}
                    </Text>
                    <List.Root color="fg.muted" gap="1" ps="4" textStyle="md">
                      {items.map((item, index) => (
                        // The catalog order is fixed for a build, so the index is a stable key.
                        <List.Item key={index}>{item}</List.Item>
                      ))}
                    </List.Root>
                  </Stack>
                ) : null}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer borderColor="border.subtle" borderTopWidth="1px" px="6" py="3">
              <Button asChild variant="outline">
                <Link href={DOCS_URL} rel="noreferrer" target="_blank">
                  <BookOpenIcon aria-hidden />
                  {t('whatsNew.readTheDocs')}
                </Link>
              </Button>
              <Button asChild variant="solid">
                <Link href={getReleaseNotesUrl(version)} rel="noreferrer" target="_blank">
                  <ScrollTextIcon aria-hidden />
                  {t('whatsNew.readReleaseNotes')}
                </Link>
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
