import type { WorkflowLibraryEntry } from '@features/workflow/data/libraryBrowseStore';

import { Badge, Box, Flex, HStack, Icon, Image, Skeleton, Stack, Text } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ImageOffIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * One workflow in the library grid: a sample-output thumbnail over a footer
 * strip of the facts that decide whether this is the workflow you want — its
 * base architecture, its size, and how many models you'd have to install to
 * run it.
 *
 * A single click selects (the right rail follows the selection); a double
 * click opens. Enrichment fills the footer in asynchronously per entry, so
 * the strip reserves its height from the first paint and never reflows the
 * grid as counts arrive.
 */

const CARD_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const CARD_FOCUS_VISIBLE = { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' } as const;
const CARD_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';
const THUMBNAIL_ASPECT_RATIO = 3 / 2;

export interface WorkflowLibraryCardProps {
  entry: WorkflowLibraryEntry;
  isSelected: boolean;
  /** Models this workflow needs that are not installed; 0 hides the badge. */
  missingCount: number;
  onOpen: (workflowId: string) => void;
  onSelect: (workflowId: string) => void;
}

export const WorkflowLibraryCard = ({
  entry,
  isSelected,
  missingCount,
  onOpen,
  onSelect,
}: WorkflowLibraryCardProps) => {
  const { t } = useTranslation();
  const [hasThumbnailFailed, setHasThumbnailFailed] = useState(false);
  const { enrichment, item } = entry;
  const workflowId = item.workflow_id;

  const handleSelect = useCallback(() => onSelect(workflowId), [onSelect, workflowId]);
  const handleOpen = useCallback(() => onOpen(workflowId), [onOpen, workflowId]);
  const handleThumbnailError = useCallback(() => setHasThumbnailFailed(true), []);

  // A broken <img> reads worse than the glyph, so a load failure falls back to
  // the same placeholder a workflow that has never run gets.
  const showThumbnail = Boolean(item.thumbnail_url) && !hasThumbnailFailed;
  const primaryBase = enrichment.status === 'ready' ? enrichment.requirements.primaryBase : null;

  return (
    <Box
      as="button"
      aria-pressed={isSelected}
      bg={isSelected ? 'bg.emphasized' : 'bg.subtle'}
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth="1px"
      cursor="pointer"
      data-workflow-card={workflowId}
      minW="0"
      overflow="hidden"
      rounded="lg"
      textAlign="start"
      transition={CARD_TRANSITION}
      w="full"
      _focusVisible={CARD_FOCUS_VISIBLE}
      _hover={CARD_HOVER}
      onClick={handleSelect}
      onDoubleClick={handleOpen}
    >
      <Box aspectRatio={THUMBNAIL_ASPECT_RATIO} bg="bg.muted" overflow="hidden" w="full">
        {showThumbnail ? (
          <Image
            alt=""
            h="full"
            objectFit="cover"
            src={item.thumbnail_url ?? undefined}
            w="full"
            onError={handleThumbnailError}
          />
        ) : (
          <Flex align="center" direction="column" gap="1" h="full" justify="center" w="full">
            <Icon aria-hidden as={ImageOffIcon} boxSize="5" color="fg.subtle" opacity={0.6} />
            <Text color="fg.subtle" fontSize="2xs">
              {t('workflowLibrary.notRunYet')}
            </Text>
          </Flex>
        )}
      </Box>
      <Stack gap="1" minW="0" p="2.5" w="full">
        <MiddleTruncate fontSize="xs" fontWeight="600" minW="0" text={item.name || t('workflowLibrary.untitled')} />
        <HStack gap="1.5" h="4" minW="0">
          {enrichment.status === 'pending' ? (
            // Enrichment in flight. An unreadable workflow ('error') gets no
            // placeholder and no error styling — the facts simply stay absent.
            <Skeleton data-enrichment-placeholder h="3" rounded="sm" variant="pulse" w="14" />
          ) : null}
          {primaryBase ? (
            <Badge flexShrink={0} size="xs" variant="subtle">
              {getModelBaseLabel(primaryBase)}
            </Badge>
          ) : null}
          {enrichment.status === 'ready' ? (
            <Text color="fg.muted" fontSize="2xs" truncate>
              {t('workflowLibrary.nodeCount', { count: enrichment.nodeCount })}
            </Text>
          ) : null}
          {missingCount > 0 ? (
            <Badge colorPalette="orange" flexShrink={0} size="xs" variant="subtle">
              {t('workflowLibrary.installModels', { count: missingCount })}
            </Badge>
          ) : null}
        </HStack>
      </Stack>
    </Box>
  );
};
