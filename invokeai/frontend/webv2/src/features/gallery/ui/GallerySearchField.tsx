import type { DateTokenParse } from '@platform/search/dateTokens';

import { Box, Icon, Input } from '@chakra-ui/react';
import { parseDateTokens } from '@platform/search/dateTokens';
import { InputShell } from '@platform/ui/InputShell';
import { SearchIcon, SparklesIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, type KeyboardEvent, type ReactNode, type Ref } from 'react';

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

const PLACEHOLDER_PROPS = { color: 'fg.subtle' } as const;

const SEARCH_START_ELEMENT = <Icon as={SearchIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />;
const SEMANTIC_START_ELEMENT = <Icon as={SparklesIcon} boxSize="3.5" color="fg.warning" flexShrink={0} />;

const MIRROR_CHIP_CSS = {
  borderRadius: 'sm',
  marginInline: '-0.5',
  paddingInline: '0.5',
} as const;

/**
 * Share a zero-padded cell to align date chips with transparent input text. Semantic descriptions remain plain
 * text with persistent mode tint.
 */
export const GallerySearchField = ({
  ariaLabel,
  describedById,
  endElement,
  inputProps,
  isInvalid,
  mode = 'metadata',
  placeholder,
  ref,
  value,
  onChange,
  onKeyDown,
}: {
  ariaLabel: string;
  describedById?: string;
  endElement?: ReactNode;
  /** Listbox wiring (`aria-controls`, `aria-activedescendant`) for hosts that drive a list from the field. */
  inputProps?: Pick<React.ComponentProps<typeof Input>, 'aria-activedescendant' | 'aria-controls' | 'role'>;
  isInvalid?: boolean;
  mode?: 'metadata' | 'semantic';
  placeholder: string;
  ref?: Ref<HTMLInputElement>;
  value: string;
  onChange: (value: string) => void;
  onKeyDown?: (event: KeyboardEvent<HTMLInputElement>) => void;
}) => {
  const mirrorRef = useRef<HTMLDivElement>(null);
  const isSemantic = mode === 'semantic';
  const segments = useMemo(
    () =>
      isSemantic ? [{ kind: 'text', text: value } as const] : getGallerySearchSegments(value, parseDateTokens(value)),
    [isSemantic, value]
  );

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
    <InputShell
      aria-invalid={isInvalid || undefined}
      data-mode={mode}
      endElement={endElement}
      position="relative"
      startElement={isSemantic ? SEMANTIC_START_ELEMENT : SEARCH_START_ELEMENT}
      tone={isSemantic ? 'warning' : undefined}
    >
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
        ref={ref}
        {...inputProps}
        aria-describedby={describedById}
        aria-invalid={isInvalid || undefined}
        aria-label={ariaLabel}
        bg="transparent"
        border="0"
        borderRadius="0"
        caretColor="fg"
        color="transparent"
        fontSize="xs"
        h="7"
        minW="0"
        px="0"
        _placeholder={PLACEHOLDER_PROPS}
        placeholder={placeholder}
        position="relative"
        w="full"
        value={value}
        onChange={handleChange}
        onKeyDown={onKeyDown}
        onScroll={handleScroll}
      />
    </InputShell>
  );
};
