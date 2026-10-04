import type { TFunction } from 'i18next';
import type { ComponentType } from 'react';

export type SettingsText = string | ((t: TFunction) => string);
export const resolveSettingsText = (text: SettingsText, t: TFunction): string =>
  typeof text === 'function' ? text(t) : text;

interface SettingBase {
  id: string;
  label: SettingsText;
  description?: SettingsText;
  group?: SettingsText;
  keywords?: string;
  scope: 'preference' | 'project' | 'instance' | 'server' | 'none';
}

export type SettingDefinition = SettingBase &
  (
    | { kind: 'boolean' }
    | { kind: 'select'; options: readonly { label: SettingsText; value: string }[] }
    | { kind: 'number' | 'slider'; min: number; max: number; step?: number }
    /** `fill`: the editor owns its scrolling and takes the dialog body's full height. */
    | { kind: 'custom'; fill?: boolean }
  );

export interface SettingsTarget {
  projectId: string;
  instanceId?: string;
}

export interface SettingFieldProps {
  field: SettingDefinition;
  surface: 'quick' | 'dialog';
  target?: SettingsTarget;
  /** Shows another setting on the surface hosting this one (the dialog or the Launchpad page). */
  onReveal?: (sectionId: string, entryId: string) => void;
}

export interface SettingsContribution {
  id: string;
  label: SettingsText;
  fields: readonly SettingDefinition[];
  /** Ordered subset of field ids. Omitted fields remain available in the dialog. */
  quick?: readonly string[];
  load: () => Promise<{ Field: ComponentType<SettingFieldProps> }>;
}
