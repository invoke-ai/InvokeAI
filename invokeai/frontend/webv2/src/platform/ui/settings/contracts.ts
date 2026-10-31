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
  /**
   * Where the value lives. `project`: layered, a project's own value (its pin) over the user's default over the
   * built-in; edited as a pin inside a project and as the default outside one.
   */
  scope: 'preference' | 'project' | 'server' | 'none';
}

export type SettingDefinition = SettingBase &
  (
    | { kind: 'boolean' }
    | { kind: 'select'; options: readonly { label: SettingsText; value: string }[] }
    | { kind: 'number' | 'slider'; min: number; max: number; step?: number }
    /**
     * `fill`: the editor owns its scrolling and takes the dialog body's full height. `framed`: the editor draws its own
     * frame (a bordered card), so the entry adds no divider under it.
     */
    | { kind: 'custom'; fill?: boolean; framed?: boolean }
  );

export interface SettingsTarget {
  projectId: string;
  instanceId?: string;
}

/**
 * Where a row's value comes from, when it is its own: the indicator after the label shows it and offers the actions
 * given. Absent: the row follows its default (or, for a plain preference, holds the built-in).
 */
export type SettingLayer =
  | {
      /** This project's own value. */
      state: 'pinned';
      /** The menu's note on what following the default gives, such as "Your default: Aspect". */
      defaultValueLabel: string;
      /** Copy the pin to the user's default, then clear the pin. */
      onUseAsDefault: () => void;
      /** Clear the pin. */
      onFollowDefault: () => void;
    }
  | {
      /** Changed from the built-in (a layered default in the Launchpad, or a plain preference). */
      state: 'default';
      /** Back to the built-in; absent where the row has its own way back, so the dot only marks the change. */
      onResetToBuiltIn?: () => void;
    };

export interface SettingFieldProps {
  field: SettingDefinition;
  surface: 'quick' | 'dialog';
  /**
   * Undefined: the field edits the user's default (the Launchpad, or a surface without a project or widget to pin
   * to). Present: it edits that project's pin, and is disabled while the target is not current.
   */
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
