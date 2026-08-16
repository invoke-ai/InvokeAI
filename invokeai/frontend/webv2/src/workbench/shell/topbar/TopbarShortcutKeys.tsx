import { chakra } from '@chakra-ui/react';
import { ShortcutKeyGlyph } from '@workbench/hotkeys/keyGlyphs';
import { IS_MAC_OS } from '@workbench/hotkeys/keys';
import { Fragment } from 'react';

import { formatTopbarShortcutPart } from './useTopbarShortcut';

/**
 * The per-key body of a topbar shortcut hint, rendered inside a `Kbd`: icons
 * where a key has a universal glyph, the platform's text label otherwise.
 * macOS runs keys together like its menu hints; elsewhere keys join with `+`.
 */
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
