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
  | { kind: 'widget'; typeId: WidgetTypeId }
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
  unavailableReason?: string;
  implemented?: boolean;
  source?: WidgetContributionSource;
}

export interface RegisteredHotkey extends HotkeyDefinition {
  keys: string[];
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
