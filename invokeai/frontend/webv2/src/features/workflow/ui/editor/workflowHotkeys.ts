/**
 * The editor's commands and the default keys it registers for them. Object entries so the translation-key scan checks
 * each `titleKey`; the defaults mirror the Workbench hotkey catalog, which the parity test holds them to.
 */
export const WORKFLOW_HOTKEYS = [
  { defaultKeys: ['shift+a', 'space'], id: 'workflows.addNode', titleKey: 'widgets.workflow.commands.addNode' },
  { defaultKeys: ['mod+c'], id: 'workflows.copySelection', titleKey: 'widgets.workflow.commands.copySelection' },
  { defaultKeys: ['mod+v'], id: 'workflows.pasteSelection', titleKey: 'widgets.workflow.commands.pasteSelection' },
  {
    defaultKeys: ['mod+shift+v'],
    id: 'workflows.pasteSelectionWithEdges',
    titleKey: 'widgets.workflow.commands.pasteSelectionWithEdges',
  },
  {
    defaultKeys: ['mod+d'],
    id: 'workflows.duplicateSelection',
    titleKey: 'widgets.workflow.commands.duplicateSelection',
  },
  { defaultKeys: ['mod+a'], id: 'workflows.selectAll', titleKey: 'widgets.workflow.commands.selectAll' },
  {
    defaultKeys: ['delete', 'backspace'],
    id: 'workflows.deleteSelection',
    titleKey: 'widgets.workflow.commands.deleteSelection',
  },
  { defaultKeys: ['mod+z'], id: 'workflows.undo', titleKey: 'widgets.workflow.commands.undo' },
  { defaultKeys: ['mod+shift+z', 'mod+y'], id: 'workflows.redo', titleKey: 'widgets.workflow.commands.redo' },
] as const;
