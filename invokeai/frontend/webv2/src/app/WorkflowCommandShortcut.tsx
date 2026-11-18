import { ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { useShortcutBinding } from '@workbench/hotkeys/useShortcutBinding';

/** A catalog command's first effective binding, remaps included, as the command palette's keycaps. */
export const WorkflowCommandShortcut = ({ commandId }: { commandId: string }) => {
  const binding = useShortcutBinding(commandId);

  return binding ? <ShortcutKeycaps parts={binding.parts} /> : null;
};
