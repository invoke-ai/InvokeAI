import { chakra } from '@chakra-ui/react';
import { ShortcutKeyGlyph } from '@workbench/hotkeys/keyGlyphs';
import { IS_MAC_OS } from '@workbench/hotkeys/keys';
import { Fragment } from 'react';

import { formatTopbarShortcutPart } from './useTopbarShortcut';

/** Use universal key glyphs or platform text; join macOS hints directly and other platforms with +. */
export const TopbarShortcutKeys = ({ parts }: { parts: string[] }) => (
  <chakra.span alignItems="center" display="inline-flex" gap="0.5">
    {parts.map((part, index) => (
      <Fragment key={`${part}:${index}`}>
        {index > 0 && !IS_MAC_OS ? <chakra.span>+</chakra.span> : null}
        <ShortcutKeyGlyph fallback={formatTopbarShortcutPart(part)} part={part} />
      </Fragment>
    ))}
  </chakra.span>
);
