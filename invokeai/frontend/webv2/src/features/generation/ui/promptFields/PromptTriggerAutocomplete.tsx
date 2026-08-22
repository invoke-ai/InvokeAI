import type { CaretRect } from '@features/generation/ui/promptFields/promptCaret';
import type {
  PromptTriggerOption,
  PromptTriggerOptionGroup,
} from '@features/generation/ui/promptFields/promptTriggerOptions';
import type { MouseEvent } from 'react';

import { Box, Portal, Stack, Text } from '@chakra-ui/react';
import { groupPromptTriggerOptions } from '@features/generation/ui/promptFields/promptTriggerOptions';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useCallback, useLayoutEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const LIST_WIDTH_PX = 260;
const MAX_LIST_HEIGHT_PX = 220;
const CARET_GAP_PX = 4;
const VIEWPORT_MARGIN_PX = 8;
const OPTION_HOVER_CSS = { bg: 'bg.emphasized' };

export const PromptTriggerAutocomplete = ({
  activeIndex,
  caretRect,
  listboxId,
  onSelect,
  options,
  optionIdPrefix,
}: {
  activeIndex: number;
  caretRect: CaretRect;
  listboxId: string;
  optionIdPrefix: string;
  options: readonly PromptTriggerOption[];
  onSelect: (option: PromptTriggerOption) => void;
}) => {
  const { t } = useTranslation();
  const listRef = useRef<HTMLDivElement | null>(null);
  const groups = useMemo(
    () =>
      groupPromptTriggerOptions(options).reduce<(PromptTriggerOptionGroup & { startIndex: number })[]>(
        (numbered, group) => {
          const previous = numbered[numbered.length - 1];

          numbered.push({ ...group, startIndex: previous ? previous.startIndex + previous.options.length : 0 });

          return numbered;
        },
        []
      ),
    [options]
  );

  const spaceBelow = window.innerHeight - (caretRect.y + caretRect.height);
  const height = Math.max(
    0,
    Math.min(MAX_LIST_HEIGHT_PX, Math.max(spaceBelow, caretRect.y) - VIEWPORT_MARGIN_PX - CARET_GAP_PX)
  );
  const opensBelow = spaceBelow >= height + CARET_GAP_PX + VIEWPORT_MARGIN_PX;
  const left = Math.max(
    VIEWPORT_MARGIN_PX,
    Math.min(caretRect.x, window.innerWidth - LIST_WIDTH_PX - VIEWPORT_MARGIN_PX)
  );
  const top = opensBelow ? caretRect.y + caretRect.height + CARET_GAP_PX : caretRect.y - height - CARET_GAP_PX;

  useLayoutEffect(() => {
    listRef.current
      ?.querySelector(`#${CSS.escape(`${optionIdPrefix}${activeIndex}`)}`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, optionIdPrefix]);

  return (
    <Portal>
      <Box
        aria-label={t('widgets.generate.completionSuggestions')}
        bg="bg.muted"
        borderColor="border.emphasized"
        borderRadius="md"
        borderWidth="1px"
        boxShadow="md"
        id={listboxId}
        left={`${left}px`}
        maxH={`${height}px`}
        overflowY="auto"
        position="fixed"
        py="1"
        ref={listRef}
        role="listbox"
        top={`${top}px`}
        w={`${LIST_WIDTH_PX}px`}
        zIndex="popover"
      >
        <Stack gap="1">
          {groups.map((group) => (
            <Stack gap="0" key={group.group}>
              <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="2" textTransform="uppercase" truncate>
                {group.group}
              </Text>
              {group.options.map((option, position) => {
                const index = group.startIndex + position;

                return (
                  <AutocompleteOption
                    key={`${option.group}-${option.value}`}
                    id={`${optionIdPrefix}${index}`}
                    isActive={index === activeIndex}
                    option={option}
                    onSelect={onSelect}
                  />
                );
              })}
            </Stack>
          ))}
        </Stack>
      </Box>
    </Portal>
  );
};

const AutocompleteOption = ({
  id,
  isActive,
  onSelect,
  option,
}: {
  id: string;
  isActive: boolean;
  option: PromptTriggerOption;
  onSelect: (option: PromptTriggerOption) => void;
}) => {
  const handleMouseDown = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }

      event.preventDefault();
      onSelect(option);
    },
    [onSelect, option]
  );

  return (
    <Box
      aria-selected={isActive}
      bg={isActive ? 'bg.emphasized' : undefined}
      color="fg"
      cursor="pointer"
      fontSize="xs"
      id={id}
      px="2"
      py="1"
      role="option"
      tabIndex={-1}
      _hover={OPTION_HOVER_CSS}
      onMouseDown={handleMouseDown}
    >
      <MiddleTruncate as="span" text={option.label} />
    </Box>
  );
};
