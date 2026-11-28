import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetContributionSource, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

export type HotkeyCategory = 'app' | 'canvas' | 'cull' | 'gallery' | 'queue' | 'viewer' | 'workflows';

/** A mode a widget turns on (`runtime.hotkeys.activateMode`); its bindings answer only in a widget that has it on. */
export type HotkeyMode = string;

export type HotkeyScope =
  | { kind: 'global' }
  /** Matches while a region holds focus: any, one named docked region, or one floating instance. */
  | { kind: 'focused-region'; region?: WidgetRegion; floatingInstanceId?: WidgetInstanceId }
  /** Matches while the key press is aimed at a widget of one of these types. */
  | { kind: 'widget'; typeIds: readonly WidgetTypeId[] }
  | { kind: 'instance'; instanceId: WidgetInstanceId };

export interface HotkeyDefinition {
  id: string;
  category: HotkeyCategory;
  commandId: string;
  defaultKeys: string[];
  title: string;
  description?: string;
  scope: HotkeyScope;
  preventDefault?: boolean;
  allowInEditable?: boolean;
  allowInModal?: boolean;
  /** A held key's auto-repeat does not run the command again: for commands that act once per press, such as a cancel. */
  ignoreRepeat?: boolean;
  /** Claim the key only while the press targets an element matching this selector; elsewhere it keeps its meaning. */
  targetSelector?: string;
  /**
   * Binds only for presses aimed at a widget with the mode on, and then outranks every other binding of its scope
   * priority; never inside a dialog or popover.
   */
  mode?: HotkeyMode;
  /**
   * Also answers with Shift held, handing its command a `HotkeyPress`, unless another binding the press can reach names
   * that Shift combination itself.
   */
  acceptsShift?: boolean;
  unavailableReason?: string;
  source?: WidgetContributionSource;
}

export interface RegisteredHotkey extends HotkeyDefinition {
  keys: string[];
  /** The user's own binding: at equal scope priority it beats a default on the same key. */
  customized?: boolean;
}

export interface HotkeyContext {
  /** The docked region holding focus, or `'floating'` while a floating window does. Never persisted. */
  focusedRegion: WidgetRegion | 'floating' | null;
  activeInstanceId: WidgetInstanceId | null;
  activeWidgetTypeId: WidgetTypeId | null;
  /** An interactive modal is open; only hotkeys that opt in with `allowInModal` run beneath it. */
  isModalPresent: boolean;
  /** Modes the widget the press is aimed at has on; absent means none. */
  activeModes?: readonly HotkeyMode[];
  projectId: string;
}

/** What an `acceptsShift` binding's command receives as its first argument. */
export interface HotkeyPress {
  shiftKey: boolean;
}

export type CustomHotkeys = Record<string, string[]>;
