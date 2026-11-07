import { Code, HStack, Icon, Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { CircleHelpIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const HELP_POSITIONING = { placement: 'bottom-end' } as const;

/** The `key:value` forms `parseDateTokens` accepts, as shown in the help popover. */
const DATE_TOKEN_EXAMPLES = [
  { descriptionKey: 'widgets.gallery.searchHelpFrom', key: 'from:2026-07-14' },
  { descriptionKey: 'widgets.gallery.searchHelpTo', key: 'to:yesterday' },
  { descriptionKey: 'widgets.gallery.searchHelpDate', key: 'date:today' },
  { descriptionKey: 'widgets.gallery.searchHelpRelative', key: 'from:7d' },
] as const;

/** What the search box does: plain text, the closed date-token grammar, and dropping an image to find similar ones. */
export const GallerySearchHelp = () => {
  const { t } = useTranslation();
  const triggerIds = useTooltipTriggerIds();

  return (
    <Popover.Root ids={triggerIds} positioning={HELP_POSITIONING}>
      <Tooltip content={t('widgets.gallery.searchHelpTitle')} ids={triggerIds}>
        <Popover.Trigger asChild>
          <IconButton aria-label={t('widgets.gallery.searchHelpTitle')} color="fg.subtle" size="sm" variant="ghost">
            <Icon as={CircleHelpIcon} boxSize="3.5" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent maxW="18rem" p="3">
            <Stack gap="2">
              <Text fontSize="md" fontWeight="600">
                {t('widgets.gallery.searchHelpTitle')}
              </Text>
              <Text color="fg.muted" fontSize="xs">
                {t('widgets.gallery.searchHelpIntro')}
              </Text>
              <Text color="fg.muted" fontSize="xs">
                {t('widgets.gallery.searchHelpDates')}
              </Text>
              <Stack gap="1.5">
                {DATE_TOKEN_EXAMPLES.map(({ descriptionKey, key }) => (
                  <HStack key={key} align="start" gap="2">
                    <Code flexShrink={0} fontSize="xs" px="1">
                      {key}
                    </Code>
                    <Text color="fg.muted" fontSize="xs">
                      {t(descriptionKey)}
                    </Text>
                  </HStack>
                ))}
              </Stack>
              <Text color="fg.muted" fontSize="xs">
                {t('widgets.gallery.searchHelpDrop')}
              </Text>
            </Stack>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
