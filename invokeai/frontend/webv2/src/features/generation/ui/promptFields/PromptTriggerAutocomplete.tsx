import type { CaretRect } from '@features/generation/ui/promptFields/promptCaret';
import type {
  PromptTriggerOption,
  PromptTriggerOptionGroup,
} from '@features/generation/ui/promptFields/promptTriggerOptions';
import type { MouseEvent } from 'react';

import { Box, Portal, Stack, Text } from '@chakra-ui/react';
import { groupPromptTriggerOptions } from '@features/generation/ui/promptFields/promptTriggerOptions';
import { useCallback, useLayoutEffect, useMemo, useRef } from 'react';

const LIST_WIDTH_PX = 260;
const MAX_LIST_HEIGHT_PX = 220;
/** Clear of the current line, so the list never covers what is being typed. */
const CARET_GAP_PX = 4;
const VIEWPORT_MARGIN_PX = 8;
const OPTION_HOVER_CSS = { bg: 'bg.emphasized' };

/**
 * The list that follows the caret while a `__` or `<` is being typed.
 *
 * Deliberately not a `Popover`: this must never take focus. The user is typing
 * in the textarea and the textarea stays the active element throughout — arrow
 * keys, Enter and Escape are handled there and only reach here as the
 * `activeIndex` prop. That also makes it a plain positioned surface rather than
 * a dismissable layer, which is why there is no trigger and no close button.
 *
 * The old picker anchored to the whole textarea and opened `bottom-start`, so in
 * a tall prompt box the list appeared at the bottom-left corner — often hundreds
 * of pixels from where the user was looking.
 */
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
  const listRef = useRef<HTMLDivElement | null>(null);
  // Each group carries where it starts in the flat list. The hook selects with
  // `options[activeIndex]`, so the ids have to be numbered the same way — and
  // counting them up as the rows render, which is what this replaced, only
  // agreed with that while grouping happened to be an order-preserving no-op.
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

  // Below the caret when there is room, above it when there is not, and never
  // off the right edge of a narrow window.
  const spaceBelow = window.innerHeight - (caretRect.y + caretRect.height);
  // Clamped: in a short viewport with the caret near the top there is room on
  // neither side, and the subtraction went negative — `maxH="-2px"` is dropped
  // by the browser, so the list sprang to full height and was then positioned
  // from that negative number.
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

  // Keyboard navigation moves a highlight the mouse never touches, so the list
  // has to follow it rather than waiting to be scrolled.
  useLayoutEffect(() => {
    listRef.current
      ?.querySelector(`#${CSS.escape(`${optionIdPrefix}${activeIndex}`)}`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [activeIndex, optionIdPrefix]);

  return (
    <Portal>
      <Box
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
  // On mousedown, not click, and with the default prevented: letting the press
  // blur the textarea would close the list out from under the pointer. Primary
  // button only — a right-click is after the context menu, not an insertion.
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
      // Focus stays in the textarea and moves through `aria-activedescendant`;
      // the options are only ever reachable that way, never by tabbing.
      tabIndex={-1}
      truncate
      _hover={OPTION_HOVER_CSS}
      onMouseDown={handleMouseDown}
    >
      {option.label}
    </Box>
  );
};
