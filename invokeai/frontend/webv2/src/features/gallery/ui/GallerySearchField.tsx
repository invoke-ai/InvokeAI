import type { DateTokenParse } from '@platform/search/dateTokens';

import { Box, Icon, Input } from '@chakra-ui/react';
import { parseDateTokens } from '@platform/search/dateTokens';
import { formControlInteraction } from '@theme/recipes';
import { SearchIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, type ReactNode } from 'react';

/** `key:value` runs the date grammar recognizes, matched against the raw value. */
const TOKEN_PATTERN = /(?:^|\s)(?:from|to|date):\S*/gi;

export interface GallerySearchSegment {
  kind: 'chip' | 'text';
  /** Chips only: the grammar rejected this value, so it filters nothing. */
  isInvalid?: boolean;
  text: string;
}

/**
 * Splits the value into plain runs and token chips. Concatenating the segments
 * must reproduce the value exactly — a dropped character shifts every glyph
 * after it out of register with the input behind.
 */
export const getGallerySearchSegments = (value: string, parse: DateTokenParse): GallerySearchSegment[] => {
  const invalidRaw = new Set(parse.invalidTokens.map((token) => `${token.key}:${token.raw}`.toLowerCase()));
  const segments: GallerySearchSegment[] = [];
  let lastIndex = 0;

  for (const match of value.matchAll(TOKEN_PATTERN)) {
    const matchIndex = match.index;
    const raw = match[0];
    // Keep the separating space out of the chip.
    const leadingSpace = raw.startsWith(' ') ? ' ' : '';
    const token = raw.slice(leadingSpace.length);

    if (matchIndex > lastIndex) {
      segments.push({ kind: 'text', text: value.slice(lastIndex, matchIndex) });
    }

    if (leadingSpace) {
      segments.push({ kind: 'text', text: leadingSpace });
    }

    segments.push({ isInvalid: invalidRaw.has(token.toLowerCase()), kind: 'chip', text: token });
    lastIndex = matchIndex + raw.length;
  }

  if (lastIndex < value.length) {
    segments.push({ kind: 'text', text: value.slice(lastIndex) });
  }

  return segments;
};

const FIELD_CSS = {
  ...formControlInteraction,
  _focusWithin: { borderColor: 'accent.solid', outline: '1px solid {colors.accent.solid}' },
} as const;

const PLACEHOLDER_PROPS = { color: 'fg.subtle' } as const;

const MIRROR_CHIP_CSS = {
  borderRadius: 'sm',
  marginInline: '-0.5',
  paddingInline: '0.5',
} as const;

/**
 * Search input that draws date tokens as chips. The mirror and the input share
 * one zero-padded cell, so they align by construction rather than by matching
 * padding; the input's own text is transparent and only its caret shows.
 */
export const GallerySearchField = ({
  ariaLabel,
  describedById,
  endElement,
  isInvalid,
  placeholder,
  value,
  onChange,
}: {
  ariaLabel: string;
  describedById?: string;
  endElement?: ReactNode;
  isInvalid?: boolean;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
}) => {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const segments = useMemo(() => getGallerySearchSegments(value, parseDateTokens(value)), [value]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => onChange(event.currentTarget.value),
    [onChange]
  );

  // The mirror has no scrollbar of its own; it follows the input's.
  const handleScroll = useCallback((event: React.UIEvent<HTMLInputElement>) => {
    const mirror = mirrorRef.current;

    if (mirror) {
      mirror.style.transform = `translateX(${String(-event.currentTarget.scrollLeft)}px)`;
    }
  }, []);

  return (
    <Box
      alignItems="center"
      aria-invalid={isInvalid || undefined}
      bg="bg"
      borderColor="border.emphasized"
      borderWidth="1px"
      css={FIELD_CSS}
      display="flex"
      gap="1.5"
      h="8"
      minW="0"
      position="relative"
      px="2"
      rounded="md"
      w="full"
    >
      <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
      <Box flex="1" minW="0" position="relative">
        {/* Flex-centred: Chakra reads a scale number in `lineHeight` as a
            unitless multiplier, which drops the text out of the field. */}
        <Box
          ref={mirrorRef}
          alignItems="center"
          aria-hidden="true"
          display="flex"
          fontSize="xs"
          inset="0"
          pointerEvents="none"
          position="absolute"
          whiteSpace="pre"
        >
          {segments.map((segment, index) =>
            segment.kind === 'chip' ? (
              <Box
                key={`${segment.text}-${String(index)}`}
                as="span"
                bg={segment.isInvalid ? 'bg.error' : 'accent.subtle'}
                color={segment.isInvalid ? 'fg.error' : 'accent.fg'}
                css={MIRROR_CHIP_CSS}
              >
                {segment.text}
              </Box>
            ) : (
              <Box key={`text-${String(index)}`} as="span">
                {segment.text}
              </Box>
            )
          )}
        </Box>
        <Input
          aria-describedby={describedById}
          aria-invalid={isInvalid || undefined}
          aria-label={ariaLabel}
          bg="transparent"
          border="0"
          borderRadius="0"
          caretColor="fg"
          color="transparent"
          fontSize="xs"
          h="8"
          minW="0"
          px="0"
          _placeholder={PLACEHOLDER_PROPS}
          placeholder={placeholder}
          position="relative"
          w="full"
          value={value}
          onChange={handleChange}
          onScroll={handleScroll}
        />
      </Box>
      {endElement}
    </Box>
  );
};
