import { ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { useTopbarShortcutBinding } from '@workbench/shell/topbar/useTopbarShortcut';

/** A catalog command's first effective binding, remaps included, as the command palette's keycaps. */
export const WorkflowCommandShortcut = ({ commandId }: { commandId: string }) => {
  const binding = useTopbarShortcutBinding(commandId);

  return binding ? <ShortcutKeycaps parts={binding.parts} /> : null;
};
