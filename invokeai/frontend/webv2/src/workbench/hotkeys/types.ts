import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetContributionSource, WidgetInstanceId, WidgetTypeId } from '@workbench/widgetContracts';

export type HotkeyCategory = 'app' | 'canvas' | 'gallery' | 'viewer' | 'workflows';

export type HotkeyScope =
  | { kind: 'global' }
  /**
   * Matches while a region holds focus: any region, one named docked region, or — for a contribution made by a
   * floating widget, whose window is no region others share — that one floating instance.
   */
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
  /** Claim the key only while the press targets an element matching this selector; elsewhere it keeps its meaning. */
  targetSelector?: string;
  unavailableReason?: string;
  implemented?: boolean;
  source?: WidgetContributionSource;
}

export interface RegisteredHotkey extends HotkeyDefinition {
  keys: string[];
  /**
   * `keys` are the user's own binding rather than `defaultKeys`. At equal scope priority a custom binding wins the
   * key over a default one, so a saved binding does not go dead when a default later takes its key.
   */
  customized?: boolean;
}

export interface HotkeyContext {
  /** The docked region holding focus, or `'floating'` while a floating window does. Never persisted. */
  focusedRegion: WidgetRegion | 'floating' | null;
  activeInstanceId: WidgetInstanceId | null;
  activeWidgetTypeId: WidgetTypeId | null;
  /** An interactive modal is open; only hotkeys that opt in with `allowInModal` run beneath it. */
  isModalPresent: boolean;
  projectId: string;
}

export type CustomHotkeys = Record<string, string[]>;
