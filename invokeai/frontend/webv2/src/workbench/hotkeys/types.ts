import type { WidgetContributionSource, WidgetInstanceId, WidgetRegion, WidgetTypeId } from '@workbench/types';

export type HotkeyCategory = 'app' | 'canvas' | 'gallery' | 'viewer' | 'workflows';

export type HotkeyScope =
  | { kind: 'global' }
  | { kind: 'focused-region'; region?: WidgetRegion }
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
  focusedRegion: WidgetRegion | null;
  activeInstanceId: WidgetInstanceId | null;
  activeWidgetTypeId: WidgetTypeId | null;
  isModalLayerActive: boolean;
  projectId: string;
}

export type CustomHotkeys = Record<string, string[]>;
