import type { SettingDefinition, SettingLayer, SettingsTarget } from '@platform/ui/settings/contracts';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { getProjectWidgetInstance, resolveWidgetValues } from '@workbench/widgetState';
import {
  useOptionalWorkbenchCommands,
  useOptionalWorkbenchQueries,
  useOptionalWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getEffectiveProjectSettings, PROJECT_SETTINGS_OWNER } from './projectSettings';
import {
  clearScopedDefault,
  getWorkbenchPreferences,
  patchScopedDefault,
  useWorkbenchPreferenceSelector,
} from './store';

export type LayeredSettingValue = boolean | string | number | null;

const EMPTY: Readonly<Record<string, unknown>> = {};

/** A value as a settings row names it: a select's option label, On/Off, the number; `null` is "Automatic". */
export const formatSettingValue = (field: SettingDefinition, value: LayeredSettingValue, t: TFunction): string => {
  if (value === null) {
    return t('settingsDialog.layer.automatic');
  }
  if (typeof value === 'boolean') {
    return t(value ? 'common.on' : 'common.off');
  }
  if (field.kind === 'select') {
    const option = field.options.find((candidate) => candidate.value === value);
    return option ? resolveSettingsText(option.label, t) : String(value);
  }
  return String(value);
};

/** The instance a widget owner's pin lives on, or undefined while the target is not the open project's. */
const getTargetInstance = (project: Project, ownerId: string, target: SettingsTarget) => {
  if (target.projectId !== project.id) {
    return undefined;
  }
  const instance = target.instanceId
    ? project.widgetInstances[target.instanceId]
    : getProjectWidgetInstance(project, ownerId);
  return instance?.typeId === ownerId ? instance : undefined;
};

/** The project's own values for `ownerId` (instance values, or the project's settings), or null while not current. */
const getPins = (project: Project, ownerId: string, target: SettingsTarget): object | null => {
  if (ownerId === PROJECT_SETTINGS_OWNER) {
    return target.projectId === project.id ? project.settings : null;
  }
  return getTargetInstance(project, ownerId, target)?.state.values ?? null;
};

/** A pin's value; undefined where the key is not pinned. */
const getPin = (pins: object, key: string): unknown =>
  Object.hasOwn(pins, key) ? (pins as Record<string, unknown>)[key] : undefined;

/** What `ownerId` reads with in the open project: its pins over the user's defaults over the built-ins. */
const getResolvedValues = (snapshot: WorkbenchSnapshot, ownerId: string, target: SettingsTarget) => {
  const project = snapshot.activeProject;
  if (ownerId === PROJECT_SETTINGS_OWNER) {
    return getEffectiveProjectSettings(project.settings, snapshot.scopedDefaults[PROJECT_SETTINGS_OWNER]) as Readonly<
      Record<string, unknown>
    >;
  }
  return resolveWidgetValues(getTargetInstance(project, ownerId, target), snapshot.scopedDefaults);
};

/**
 * One per-project setting, layered as project pin, then the user's default, then `builtIn`. `ownerId` is the
 * `scopedDefaults` key: a widget type id, where `generate` pins the project's own generation flags rather than the
 * Generate widget's values. `read` is the owner's normalizer over a possibly partial record. `field` names values in
 * the indicator's menu.
 *
 * Without a target the row edits the user's default, and its layer is `default` while that differs from the built-in.
 * With one it edits the target's pin, fenced against a stale project, account or instance, and its layer is `pinned`
 * while the pin exists. `set` takes the value stored under `key`, which is what `read` turns back into the row's value.
 */
export const useLayeredSetting = <V extends LayeredSettingValue>(
  ownerId: string,
  key: string,
  target: SettingsTarget | undefined,
  read: (values: Readonly<Record<string, unknown>>) => V,
  builtIn: V,
  field: SettingDefinition
): { value: V; layer: SettingLayer | undefined; set: (value: V) => void; disabled: boolean } => {
  const { t } = useTranslation();
  const [accountScope] = useState(captureAccountScope);
  const commands = useOptionalWorkbenchCommands();
  const queries = useOptionalWorkbenchQueries();
  const defaults = useWorkbenchPreferenceSelector((preferences) => preferences.scopedDefaults[ownerId] ?? EMPTY);
  const defaultValue = read(defaults);
  const hasDefault = Object.hasOwn(defaults, key);
  const pinned = useOptionalWorkbenchSelector((snapshot) => {
    const pins = target ? getPins(snapshot.activeProject, ownerId, target) : null;
    return pins === null ? null : getPin(pins, key) !== undefined;
  }, null);
  const resolvedValue = useOptionalWorkbenchSelector<V | undefined>(
    (snapshot) => (target ? read(getResolvedValues(snapshot, ownerId, target)) : undefined),
    undefined
  );
  const projectId = target?.projectId;
  const instanceId = target?.instanceId;
  // Null when there is no pin to write, so a requested target that is not current leaves the row disabled.
  const isPinMode = target !== undefined;
  const disabled = isPinMode && pinned === null;
  const value = isPinMode ? (resolvedValue ?? builtIn) : defaultValue;

  const isCurrent = useCallback(
    () => isAccountScopeCurrent(accountScope) && (!projectId || queries?.isActiveProject(projectId) === true),
    [accountScope, projectId, queries]
  );
  /** The current pins for this target, read at the moment of a write. */
  const readPins = useCallback(() => {
    if (!projectId || !queries || !isCurrent()) {
      return null;
    }
    const project = queries.getProject(projectId);
    return project ? getPins(project, ownerId, { instanceId, projectId }) : null;
  }, [instanceId, isCurrent, ownerId, projectId, queries]);
  const instanceIdFor = useCallback(() => {
    const project = projectId ? queries?.getProject(projectId) : undefined;
    return project && projectId ? getTargetInstance(project, ownerId, { instanceId, projectId })?.id : undefined;
  }, [instanceId, ownerId, projectId, queries]);

  const writeDefault = useCallback(
    (stored: unknown) => {
      if (!isAccountScopeCurrent(accountScope)) {
        return;
      }
      const current = getWorkbenchPreferences().scopedDefaults[ownerId] ?? EMPTY;
      // A default equal to the built-in is no default: the key goes, so the record keeps only real choices.
      if (read({ ...current, [key]: stored }) === builtIn) {
        void clearScopedDefault(ownerId, [key]);
      } else {
        void patchScopedDefault(ownerId, { [key]: stored });
      }
    },
    [accountScope, builtIn, key, ownerId, read]
  );
  const writePin = useCallback(
    (stored: unknown) => {
      if (!commands || !projectId || readPins() === null) {
        return;
      }
      if (ownerId === PROJECT_SETTINGS_OWNER) {
        commands.account.updateProjectPreferences({ [key]: stored });
        return;
      }
      const id = instanceIdFor();
      if (id) {
        commands.widgets.patchInstanceValues(id, { [key]: stored }, projectId);
      }
    },
    [commands, instanceIdFor, key, ownerId, projectId, readPins]
  );
  const clearPin = useCallback(() => {
    if (!commands || !projectId || readPins() === null) {
      return;
    }
    if (ownerId === PROJECT_SETTINGS_OWNER) {
      commands.account.updateProjectPreferences({ [key]: undefined });
      return;
    }
    const id = instanceIdFor();
    if (id) {
      commands.widgets.clearInstanceValues(id, [key], projectId);
    }
  }, [commands, instanceIdFor, key, ownerId, projectId, readPins]);

  const set = useCallback(
    (next: V) => (isPinMode ? writePin(next) : writeDefault(next)),
    [isPinMode, writeDefault, writePin]
  );
  const promoteToDefault = useCallback(() => {
    const pins = readPins();
    const pin = pins ? getPin(pins, key) : undefined;
    if (pin === undefined) {
      return;
    }
    writeDefault(pin);
    clearPin();
  }, [clearPin, key, readPins, writeDefault]);
  const resetToBuiltIn = useCallback(() => {
    if (isAccountScopeCurrent(accountScope)) {
      void clearScopedDefault(ownerId, [key]);
    }
  }, [accountScope, key, ownerId]);

  const defaultValueText = formatSettingValue(field, defaultValue, t);
  const layer = useMemo<SettingLayer | undefined>(() => {
    if (isPinMode) {
      return pinned
        ? {
            defaultValueLabel: t(
              hasDefault ? 'settingsDialog.layer.yourDefault' : 'settingsDialog.layer.invokeDefault',
              {
                value: defaultValueText,
              }
            ),
            onFollowDefault: clearPin,
            onUseAsDefault: promoteToDefault,
            state: 'pinned',
          }
        : undefined;
    }
    return defaultValue !== builtIn ? { onResetToBuiltIn: resetToBuiltIn, state: 'default' } : undefined;
  }, [
    builtIn,
    clearPin,
    defaultValue,
    defaultValueText,
    hasDefault,
    isPinMode,
    pinned,
    resetToBuiltIn,
    t,
    promoteToDefault,
  ]);

  return { disabled, layer, set, value };
};
