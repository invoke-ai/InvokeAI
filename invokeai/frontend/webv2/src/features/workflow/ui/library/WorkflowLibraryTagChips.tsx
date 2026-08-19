import type { WorkflowTagCount } from '@features/workflow/data/libraryBrowseStore';

import { HStack, Text } from '@chakra-ui/react';
import { Button, Scrollable } from '@platform/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Single-select tag filter for the library grid. The counts come from the
 * backend for the whole category, not the loaded page, so a chip's number
 * still means "how many workflows carry this tag" after infinite scroll has
 * only fetched the first page.
 */

// A ScrollArea root defaults to `height: 100%` and grows to its content's
// max-content width, so a chip strip needs both an explicit height and an
// inline-size containment or it stretches the dialog header.
const CHIP_STRIP_CONTAIN_CSS = { contain: 'inline-size' } as const;
const CHIP_STRIP_CONTENT_PROPS = { h: 'full' } as const;

interface WorkflowLibraryTagChipsProps {
  selectedTag: string | null;
  tagCounts: readonly WorkflowTagCount[];
  onSelect: (tag: string | null) => void;
}

const TagChip = ({
  count,
  isSelected,
  label,
  tag,
  onSelect,
}: {
  count: number | null;
  isSelected: boolean;
  label: string;
  tag: string | null;
  onSelect: (tag: string | null) => void;
}) => {
  const handleClick = useCallback(() => onSelect(tag), [onSelect, tag]);

  return (
    <Button
      aria-pressed={isSelected}
      flexShrink={0}
      rounded="full"
      size="2xs"
      variant={isSelected ? 'subtle' : 'ghost'}
      onClick={handleClick}
    >
      {label}
      {count === null ? null : (
        <Text as="span" color="fg.subtle" fontSize="2xs">
          {count}
        </Text>
      )}
    </Button>
  );
};

export const WorkflowLibraryTagChips = ({ selectedTag, tagCounts, onSelect }: WorkflowLibraryTagChipsProps) => {
  const { t } = useTranslation();

  // A lone "All" chip filters nothing. The strip stays while a tag is still
  // applied, so a category whose counts failed to load can always be cleared.
  if (tagCounts.length === 0 && selectedTag === null) {
    return null;
  }

  return (
    <Scrollable
      contentProps={CHIP_STRIP_CONTENT_PROPS}
      css={CHIP_STRIP_CONTAIN_CSS}
      flexShrink={0}
      h="1.75rem"
      label={t('workflowLibrary.tagsLabel')}
      minW="0"
      orientation="horizontal"
      w="full"
    >
      <HStack align="center" gap="1" h="full" minW="0">
        <TagChip
          count={null}
          isSelected={selectedTag === null}
          label={t('workflowLibrary.allTag')}
          tag={null}
          onSelect={onSelect}
        />
        {tagCounts.map(({ count, tag }) => (
          <TagChip key={tag} count={count} isSelected={selectedTag === tag} label={tag} tag={tag} onSelect={onSelect} />
        ))}
      </HStack>
    </Scrollable>
  );
};
