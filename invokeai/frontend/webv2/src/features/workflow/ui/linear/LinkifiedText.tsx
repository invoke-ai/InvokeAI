import type { TextProps } from '@chakra-ui/react';

import { Link, Text } from '@chakra-ui/react';
import { useMemo } from 'react';

import { splitTextIntoLinkSegments } from './linkify';

/** Prose with its `http(s)` URLs rendered as links that open in a new tab; the text itself is left as written. */
export const LinkifiedText = ({ children, ...textProps }: { children: string } & Omit<TextProps, 'children'>) => {
  const segments = useMemo(() => splitTextIntoLinkSegments(children), [children]);

  return (
    <Text {...textProps}>
      {segments.map((segment, index) =>
        segment.kind === 'link' ? (
          <Link
            key={index}
            color="accent.fg"
            href={segment.value}
            overflowWrap="anywhere"
            rel="noopener noreferrer"
            target="_blank"
          >
            {segment.value}
          </Link>
        ) : (
          segment.value
        )
      )}
    </Text>
  );
};
