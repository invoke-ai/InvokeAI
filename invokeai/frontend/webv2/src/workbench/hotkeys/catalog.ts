import type { HotkeyCategory, HotkeyDefinition, HotkeyScope } from './types';

import { IS_MAC_OS } from './keys';

/**
 * Every binding the app ships, so Hotkeys settings list each under its widget whether or not that widget is mounted.
 * A widget registers the same ids with the same defaults while it is mounted (parity tests hold them together); only
 * registered handlers intercept chords. App chords are global, other chords require widget focus, and editable fields
 * are excluded unless explicitly allowed. Overrides include print/search/settings, undo/redo, selection/clipboard, and
 * Canvas merge/deselect/invert; browser-reserved chords work only when delivered to the page.
 */

const categoryScopes = {
  app: { kind: 'global' },
  canvas: { kind: 'widget', typeIds: ['canvas'] },
  gallery: { kind: 'widget', typeIds: ['gallery'] },
  viewer: { kind: 'widget', typeIds: ['preview'] },
  workflows: { kind: 'widget', typeIds: ['workflow'] },
} as const satisfies Record<HotkeyCategory, HotkeyScope>;

/**
 * Recall and workflow loading act on the gallery selection, which Preview shows, so they answer wherever the user
 * selects or views it: Gallery or Preview. Remix stays per widget because Gallery binds its own.
 */
const selectedImageScope: HotkeyScope = { kind: 'widget', typeIds: ['preview', 'gallery'] };

/**
 * Keep invocation shortcuts available inside text and numeric fields so prompt editing does not swallow
 * submission.
 */
const editableAppHotkeys = new Set([
  'app.focusPrompt',
  'app.invoke',
  'app.invokeFront',
  'app.invokeToOtherDestination',
  'app.openProjectSwitcher',
  // Settings is a global destination with no meaning inside a text field, so
  // the caret's location should not decide whether it opens.
  'app.openSettings',
  'app.promptHistoryNext',
  'app.promptHistoryPrev',
  'app.promptWeightDown',
  'app.promptWeightUp',
]);

const formatHotkeyTitle = (id: string): string =>
  id
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');

/** `title` overrides the one derived from `id` where the command was renamed but its id kept for saved bindings. */
const hotkey = (
  category: HotkeyCategory,
  id: string,
  defaultKeys: string[],
  { scope = categoryScopes[category], title = formatHotkeyTitle(id) }: { scope?: HotkeyScope; title?: string } = {}
): HotkeyDefinition => {
  const commandId = `${category}.${id}`;

  return {
    allowInEditable: editableAppHotkeys.has(commandId),
    category,
    commandId,
    defaultKeys,
    id: commandId,
    preventDefault: true,
    scope,
    title,
  };
};

// The palette must open while typing and toggle closed above its own modal
// layer, so it opts into both editable and modal contexts.
export const OPEN_COMMAND_PALETTE_HOTKEY: HotkeyDefinition = {
  allowInEditable: true,
  allowInModal: true,
  category: 'app',
  commandId: 'app.openCommandPalette',
  defaultKeys: ['mod+k'],
  id: 'app.openCommandPalette',
  preventDefault: true,
  scope: { kind: 'global' },
  title: 'Open Command Palette',
};

/**
 * Generation's `PromptTextarea` marks an editable prompt this way only while the selected model reads attention
 * syntax. Anywhere else Ctrl+Up/Down keeps its text-field meaning.
 */
const PROMPT_ATTENTION_TARGET_SELECTOR = 'textarea[data-prompt-attention-target="true"]:not([readonly])';

export type RegionFocusDirection = 'down' | 'left' | 'right' | 'up';

/**
 * Region focus stays reachable from text fields, so its chord must be one no text field binds: Alt+Shift+Arrow,
 * except on macOS, where most modified arrows are taken. Option+Shift+Arrow selects by word or paragraph,
 * Control+Arrow and Control+Shift+Arrow belong to Mission Control (and never reach the page), Control+Option+Arrow
 * is VoiceOver's navigation chord, and Command+Option+Left/Right switch browser tabs. Control+Command+Arrow has no
 * stock macOS, browser, text-field or VoiceOver binding that we know of.
 */
export const getRegionFocusDefaultKey = (direction: RegionFocusDirection, isMacOs = IS_MAC_OS): string =>
  `${isMacOs ? 'mod+ctrl' : 'alt+shift'}+arrow${direction}`;

export const regionFocusHotkeys = (['left', 'right', 'up', 'down'] as const).map((direction) => ({
  ...hotkey('app', `focusRegion${direction[0].toUpperCase()}${direction.slice(1)}`, [
    getRegionFocusDefaultKey(direction),
  ]),
  allowInEditable: true,
  direction,
}));

export const widgetCycleHotkeys = (
  [
    { direction: -1, id: 'selectPreviousWidget', key: 'alt+pageup' },
    { direction: 1, id: 'selectNextWidget', key: 'alt+pagedown' },
  ] as const
).map(({ direction, id, key }) => ({
  ...hotkey('app', id, [key]),
  allowInEditable: true,
  direction,
  scope: { kind: 'focused-region' } as const,
}));

export const firstPartyHotkeyCatalog: HotkeyDefinition[] = [
  OPEN_COMMAND_PALETTE_HOTKEY,
  ...regionFocusHotkeys,
  ...widgetCycleHotkeys,
  hotkey('app', 'invoke', ['mod+enter']),
  // Override the destination for one submission without changing the saved route.
  hotkey('app', 'invokeToOtherDestination', ['alt+mod+enter']),
  hotkey('app', 'invokeFront', ['mod+shift+enter']),
  hotkey('app', 'openProjectSwitcher', ['mod+p']),
  // Use mod for the platform settings chord: Command on macOS, Control elsewhere.
  hotkey('app', 'openSettings', ['mod+,']),
  hotkey('app', 'saveLayoutPreset', []),
  hotkey('app', 'selectComposePreset', ['alt+1']),
  hotkey('app', 'selectEditPreset', ['alt+2']),
  hotkey('app', 'selectVideoPreset', ['alt+3']),
  hotkey('app', 'selectAutomatePreset', ['alt+4'], { title: 'Select Workflow Preset' }),
  hotkey('app', 'cancelQueueItem', ['shift+x']),
  hotkey('app', 'clearQueue', ['mod+shift+x'], { title: 'Cancel all queue items' }),
  // The ids keep "Tab" so saved bindings survive; each opens a widget, wherever the layout keeps it.
  hotkey('app', 'selectGenerateTab', ['1'], { title: 'Open Generate' }),
  hotkey('app', 'selectCanvasTab', ['2'], { title: 'Open Canvas' }),
  hotkey('app', 'selectUpscalingTab', ['3'], { title: 'Open Upscale' }),
  hotkey('app', 'selectWorkflowsTab', ['4'], { title: 'Open Workflow editor' }),
  hotkey('app', 'selectModelsTab', ['5'], { title: 'Open Model Manager' }),
  hotkey('app', 'selectQueueTab', ['6'], { title: 'Open Queue' }),
  hotkey('app', 'openWidgets', [], { title: 'Show widgets' }),
  hotkey('app', 'promptHistoryPrev', ['alt+arrowup']),
  hotkey('app', 'promptHistoryNext', ['alt+arrowdown']),
  { ...hotkey('app', 'promptWeightUp', ['ctrl+arrowup']), targetSelector: PROMPT_ATTENTION_TARGET_SELECTOR },
  { ...hotkey('app', 'promptWeightDown', ['ctrl+arrowdown']), targetSelector: PROMPT_ATTENTION_TARGET_SELECTOR },
  hotkey('app', 'focusPrompt', ['alt+a']),
  hotkey('app', 'toggleLeftPanel', ['t', 'o']),
  hotkey('app', 'toggleRightPanel', ['g']),
  // Layout history: Mod+Z belongs to whichever editor holds focus, so layout steps take Alt with it.
  hotkey('app', 'undoLayoutChange', ['mod+alt+z'], { title: 'Undo layout change' }),
  hotkey('app', 'redoLayoutChange', ['mod+alt+shift+z'], { title: 'Redo layout change' }),
  // No default key: it throws away unsaved arrangement (Undo layout change brings it back), and Mod+Shift+R is the
  // browser's hard reload. The id keeps "Reset" so saved bindings survive.
  hotkey('app', 'resetPanelLayout', [], { title: 'Revert layout to saved preset' }),
  hotkey('app', 'togglePanels', ['f']),
  hotkey('app', 'togglePreview', ['z']),
  // Canvas rows mirror the Canvas widget's own tables (canvasToolCommands, canvasHotkeyCommands, canvasHeaderCommands).
  hotkey('canvas', 'tool.view', ['h'], { title: 'Select View Tool' }),
  hotkey('canvas', 'tool.move', ['v'], { title: 'Select Move Tool' }),
  hotkey('canvas', 'transformSelected', ['shift+t'], { title: 'Transform Selected Layer' }),
  // C is the Canvas's own hold-for-frame key, so the frame tool has no default binding.
  hotkey('canvas', 'tool.bbox', [], { title: 'Select Frame Tool' }),
  hotkey('canvas', 'tool.brush', ['b'], { title: 'Select Brush Tool' }),
  hotkey('canvas', 'tool.eraser', ['e'], { title: 'Select Eraser Tool' }),
  hotkey('canvas', 'tool.lasso', ['l'], { title: 'Select Lasso Tool' }),
  hotkey('canvas', 'tool.marquee', ['m'], { title: 'Select Marquee Tool' }),
  hotkey('canvas', 'tool.shape', ['u'], { title: 'Select Shape Tool' }),
  hotkey('canvas', 'tool.text', ['t'], { title: 'Select Text Tool' }),
  hotkey('canvas', 'tool.gradient', ['g'], { title: 'Select Gradient Tool' }),
  hotkey('canvas', 'tool.colorPicker', ['i'], { title: 'Select Color Picker' }),
  hotkey('canvas', 'brushSizeDown', ['['], { title: 'Decrease Brush Size' }),
  hotkey('canvas', 'brushSizeUp', [']'], { title: 'Increase Brush Size' }),
  hotkey('canvas', 'toggleFillColor', ['x'], { title: 'Swap Colors' }),
  hotkey('canvas', 'setFillColorsToDefault', ['d'], { title: 'Reset Colors' }),
  hotkey('canvas', 'undo', ['mod+z']),
  hotkey('canvas', 'redo', ['mod+shift+z', 'mod+y']),
  hotkey('canvas', 'prevEntity', ['alt+[']),
  hotkey('canvas', 'nextEntity', ['alt+]']),
  hotkey('canvas', 'quickSwitch', ['q']),
  hotkey('canvas', 'deleteSelected', ['delete', 'backspace']),
  hotkey('canvas', 'resetSelected', ['shift+c']),
  hotkey('canvas', 'toggleNonRasterLayers', ['shift+h']),
  hotkey('canvas', 'copySelection', ['mod+c']),
  hotkey('canvas', 'cutSelection', ['mod+x']),
  hotkey('canvas', 'pasteImage', ['mod+v']),
  hotkey('canvas', 'selectAll', ['mod+a']),
  hotkey('canvas', 'deselect', ['mod+d']),
  hotkey('canvas', 'invertSelection', ['mod+shift+i']),
  hotkey('canvas', 'nudgeLeft', ['arrowleft']),
  hotkey('canvas', 'nudgeRight', ['arrowright']),
  hotkey('canvas', 'nudgeUp', ['arrowup']),
  hotkey('canvas', 'nudgeDown', ['arrowdown']),
  hotkey('canvas', 'nudgeLeftLarge', ['shift+arrowleft']),
  hotkey('canvas', 'nudgeRightLarge', ['shift+arrowright']),
  hotkey('canvas', 'nudgeUpLarge', ['shift+arrowup']),
  hotkey('canvas', 'nudgeDownLarge', ['shift+arrowdown']),
  hotkey('canvas', 'duplicateLayer', ['mod+j']),
  hotkey('canvas', 'groupLayers', ['mod+g']),
  hotkey('canvas', 'ungroupLayers', ['mod+shift+g']),
  hotkey('canvas', 'mergeDown', ['mod+e']),
  hotkey('canvas', 'layerForward', ['mod+]']),
  hotkey('canvas', 'layerBackward', ['mod+[']),
  hotkey('canvas', 'layerToFront', ['mod+shift+]']),
  hotkey('canvas', 'layerToBack', ['mod+shift+[']),
  hotkey('canvas', 'cropAllLayersToFrame', []),
  hotkey('canvas', 'fitBboxToLayers', ['shift+n'], { title: 'Fit Frame To Layers' }),
  hotkey('canvas', 'fitBboxToMasks', ['shift+b'], { title: 'Fit Frame To Masks' }),
  // The gallery saves have no default keys; they are assignable through the hotkeys settings.
  hotkey('canvas', 'saveToGallery', []),
  hotkey('canvas', 'saveBboxToGallery', [], { title: 'Save Frame To Gallery' }),
  // Clear canvas keeps its saved-binding id, `newSession`.
  hotkey('canvas', 'newSession', ['mod+alt+n'], { title: 'Clear Canvas' }),
  hotkey('canvas', 'setZoomTo100Percent', ['mod+1'], { title: 'Zoom to 100%' }),
  hotkey('canvas', 'setZoomTo200Percent', ['mod+2'], { title: 'Zoom to 200%' }),
  hotkey('canvas', 'setZoomTo400Percent', ['mod+3'], { title: 'Zoom to 400%' }),
  hotkey('canvas', 'setZoomTo800Percent', ['mod+4'], { title: 'Zoom to 800%' }),
  hotkey('workflows', 'addNode', ['shift+a', 'space']),
  hotkey('workflows', 'copySelection', ['mod+c']),
  hotkey('workflows', 'pasteSelection', ['mod+v']),
  hotkey('workflows', 'pasteSelectionWithEdges', ['mod+shift+v']),
  hotkey('workflows', 'duplicateSelection', ['mod+d']),
  hotkey('workflows', 'selectAll', ['mod+a']),
  hotkey('workflows', 'deleteSelection', ['delete', 'backspace']),
  hotkey('workflows', 'undo', ['mod+z']),
  hotkey('workflows', 'redo', ['mod+shift+z', 'mod+y']),
  hotkey('viewer', 'swapImages', ['c']),
  hotkey('viewer', 'nextComparisonMode', ['m']),
  hotkey('viewer', 'deleteImage', ['delete', 'backspace']),
  hotkey('viewer', 'loadWorkflow', ['w'], { scope: selectedImageScope }),
  hotkey('viewer', 'recallAll', ['a'], { scope: selectedImageScope }),
  hotkey('viewer', 'recallSeed', ['s'], { scope: selectedImageScope }),
  hotkey('viewer', 'recallPrompts', ['p'], { scope: selectedImageScope }),
  hotkey('viewer', 'remix', ['r']),
  hotkey('viewer', 'useSize', ['d'], { scope: selectedImageScope }),
  hotkey('viewer', 'toggleMetadata', ['i'], { title: 'Toggle Details' }),
  // Preview registers these while it shows an item; the arrows step saved items outside comparison.
  hotkey('viewer', 'toggleFilmstrip', ['t']),
  hotkey('viewer', 'previousItem', ['arrowleft']),
  hotkey('viewer', 'nextItem', ['arrowright']),
  hotkey('viewer', 'zoomToActual', ['1'], { title: 'Zoom to Actual Size' }),
  hotkey('viewer', 'zoomToFit', ['f'], { title: 'Zoom to Fit' }),
  hotkey('gallery', 'selectAllOnPage', ['mod+a']),
  // No default key: Esc in the Gallery is left to Preview interactions, and the Preview label's clear button
  // deselects. Users can still assign one.
  hotkey('gallery', 'clearSelection', []),
  hotkey('gallery', 'galleryNavUp', ['up']),
  hotkey('gallery', 'galleryNavRight', ['right']),
  hotkey('gallery', 'galleryNavDown', ['down']),
  hotkey('gallery', 'galleryNavLeft', ['left']),
  hotkey('gallery', 'galleryNavUpAlt', ['alt+up']),
  hotkey('gallery', 'galleryNavRightAlt', ['alt+right']),
  hotkey('gallery', 'galleryNavDownAlt', ['alt+down']),
  hotkey('gallery', 'galleryNavLeftAlt', ['alt+left']),
  // Shift extends the selection; Ctrl (Cmd on macOS) moves focus alone. Toggling the focused item has no default
  // key: Cmd+Space is Spotlight on macOS, and Ctrl+Space already toggles by activating the focused thumbnail as a
  // Ctrl+click. Ctrl+Enter is not used: it invokes from anywhere, the gallery included.
  hotkey('gallery', 'extendSelectionUp', ['shift+up']),
  hotkey('gallery', 'extendSelectionRight', ['shift+right']),
  hotkey('gallery', 'extendSelectionDown', ['shift+down']),
  hotkey('gallery', 'extendSelectionLeft', ['shift+left']),
  hotkey('gallery', 'moveFocusUp', ['mod+up']),
  hotkey('gallery', 'moveFocusRight', ['mod+right']),
  hotkey('gallery', 'moveFocusDown', ['mod+down']),
  hotkey('gallery', 'moveFocusLeft', ['mod+left']),
  hotkey('gallery', 'toggleFocusedInSelection', []),
  hotkey('gallery', 'deleteSelection', ['delete', 'backspace']),
  hotkey('gallery', 'remix', ['r']),
  hotkey('gallery', 'starImage', ['.']),
  hotkey('gallery', 'toggleStarredOnly', []),
];
