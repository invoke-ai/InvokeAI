import type { SettingDefinition, SettingLayer, SettingsTarget } from '@platform/ui/settings/contracts';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchSnapshot } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { areJsonValuesStructurallyEqual } from '@platform/core/json';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { withdrawNotificationUndos } from '@workbench/shell/notifications/undoRegistry';
import { useNotify } from '@workbench/useNotify';
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

/**
 * The instance a widget owner's pin lives on: always the project's instance of the owner's type, the one every reader
 * of its values reads (an alias such as `gallery:center` shares it). A requested `instanceId` only fences: it must
 * still exist with the owner's type. Undefined while the target is not the open project's or fails that fence.
 */
const getTargetInstance = (project: Project, ownerId: string, target: SettingsTarget) => {
  if (target.projectId !== project.id) {
    return undefined;
  }
  if (target.instanceId && project.widgetInstances[target.instanceId]?.typeId !== ownerId) {
    return undefined;
  }
  const instance = getProjectWidgetInstance(project, ownerId);
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
  const notify = useNotify();
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
        commands.account.updateProjectPreferences({ [key]: stored }, projectId);
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
      commands.account.updateProjectPreferences({ [key]: undefined }, projectId);
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
  const settingName = resolveSettingsText(field.label, t);
  /** The user's default for `key` as it stands: whether one exists, and its stored value. */
  const readDefault = useCallback(() => {
    const current = getWorkbenchPreferences().scopedDefaults[ownerId] ?? EMPTY;
    return { exists: Object.hasOwn(current, key), value: current[key] };
  }, [key, ownerId]);
  const restoreDefault = useCallback(
    (prior: { exists: boolean; value: unknown }) =>
      void (prior.exists ? patchScopedDefault(ownerId, { [key]: prior.value }) : clearScopedDefault(ownerId, [key])),
    [key, ownerId]
  );
  /**
   * The pin an Undo would restore into `pinProjectId`, addressed by project rather than through the row's target: the
   * project may be open but not active by then, and an alias instance the row came from may be gone.
   */
  const readProjectPins = useCallback(
    (pinProjectId: string) => {
      const project = queries?.getProject(pinProjectId);
      return project ? getPins(project, ownerId, { projectId: pinProjectId }) : null;
    },
    [ownerId, queries]
  );
  /** Whether `pinProjectId` is open with `key` unpinned, the state each pin action leaves behind. */
  const isUnpinnedIn = useCallback(
    (pinProjectId: string) => {
      const pins = readProjectPins(pinProjectId);
      return pins !== null && getPin(pins, key) === undefined;
    },
    [key, readProjectPins]
  );
  /** Re-pins `stored` in `pinProjectId`, addressed by project id: it may be open but not active by then. */
  const repinIn = useCallback(
    (pinProjectId: string, stored: unknown) => {
      const project = queries?.getProject(pinProjectId);
      if (!commands || !project) {
        return;
      }
      if (ownerId === PROJECT_SETTINGS_OWNER) {
        commands.account.updateProjectPreferences({ [key]: stored }, pinProjectId);
        return;
      }
      const instance = getTargetInstance(project, ownerId, { projectId: pinProjectId });
      if (instance) {
        commands.widgets.patchInstanceValues(instance.id, { [key]: stored }, pinProjectId);
      }
    },
    [commands, key, ownerId, queries]
  );
  /**
   * Reports an action with an Undo. `restore` restores the setting and returns true while it still holds what the
   * action left; false (changed since) says there is nothing to undo.
   */
  const offerUndo = useCallback(
    (title: string, options: { projectId?: string; supersedes: string }, restore: () => boolean) =>
      notify.undoable(title, {
        onUndo: () => {
          if (isAccountScopeCurrent(accountScope) && !restore()) {
            notify.info(t('settingsDialog.layer.undoStale', { setting: settingName }));
          }
        },
        ...(options.projectId ? { projectId: options.projectId } : {}),
        supersedes: options.supersedes,
      }),
    [accountScope, notify, settingName, t]
  );
  // A later action on the same key withdraws an earlier Undo. Defaults are account-wide, so the actions that change one
  // share an account-wide slot; following the default only changes its project.
  const defaultUndoKey = `settings-layer:${ownerId}:${key}`;
  const getFollowUndoKey = useCallback(
    (pinProjectId: string) => `settings-layer:${pinProjectId}:${ownerId}:${key}`,
    [key, ownerId]
  );
  const valueText = useCallback(
    (values: Readonly<Record<string, unknown>>) => formatSettingValue(field, read(values), t),
    [field, read, t]
  );

  const followDefault = useCallback(() => {
    const pins = readPins();
    const pin = pins ? getPin(pins, key) : undefined;
    if (pin === undefined || !projectId) {
      return;
    }
    clearPin();
    const defaults = getWorkbenchPreferences().scopedDefaults[ownerId] ?? EMPTY;
    const title = t(
      Object.hasOwn(defaults, key)
        ? 'settingsDialog.layer.followDefaultDone'
        : 'settingsDialog.layer.followInvokeDefaultDone',
      { setting: settingName, value: valueText(defaults) }
    );
    offerUndo(title, { projectId, supersedes: getFollowUndoKey(projectId) }, () => {
      if (!isUnpinnedIn(projectId)) {
        return false;
      }
      repinIn(projectId, pin);
      return true;
    });
  }, [
    clearPin,
    getFollowUndoKey,
    isUnpinnedIn,
    key,
    offerUndo,
    ownerId,
    projectId,
    readPins,
    repinIn,
    settingName,
    t,
    valueText,
  ]);
  const promoteToDefault = useCallback(() => {
    const pins = readPins();
    const pin = pins ? getPin(pins, key) : undefined;
    if (pin === undefined || !projectId) {
      return;
    }
    const prior = readDefault();
    const written = { ...(getWorkbenchPreferences().scopedDefaults[ownerId] ?? EMPTY), [key]: pin };
    // `writeDefault` drops a default equal to the built-in, so the default it leaves is either the pin or nothing.
    const writesDefault = read(written) !== builtIn;
    writeDefault(pin);
    clearPin();
    const isDefaultAsWritten = () => {
      const current = readDefault();
      return writesDefault ? current.exists && areJsonValuesStructurallyEqual(current.value, pin) : !current.exists;
    };
    const title = writesDefault
      ? t('settingsDialog.layer.useAsDefaultDone', { setting: settingName, value: valueText(written) })
      : t('settingsDialog.layer.followInvokeDefaultDone', { setting: settingName, value: valueText(EMPTY) });
    // An earlier Follow Undo here would re-pin over the new default, leaving this Undo nothing to restore.
    withdrawNotificationUndos(getFollowUndoKey(projectId));
    offerUndo(title, { projectId, supersedes: defaultUndoKey }, () => {
      if (!isDefaultAsWritten() || !isUnpinnedIn(projectId)) {
        return false;
      }
      restoreDefault(prior);
      repinIn(projectId, pin);
      return true;
    });
  }, [
    builtIn,
    clearPin,
    getFollowUndoKey,
    defaultUndoKey,
    isUnpinnedIn,
    key,
    offerUndo,
    ownerId,
    projectId,
    read,
    readDefault,
    readPins,
    repinIn,
    restoreDefault,
    settingName,
    t,
    valueText,
    writeDefault,
  ]);
  const resetToBuiltIn = useCallback(() => {
    if (!isAccountScopeCurrent(accountScope)) {
      return;
    }
    const prior = readDefault();
    void clearScopedDefault(ownerId, [key]);
    if (!prior.exists) {
      return;
    }
    offerUndo(
      t('settingsDialog.layer.resetToBuiltInDone', { setting: settingName, value: valueText(EMPTY) }),
      { supersedes: defaultUndoKey },
      () => {
        if (readDefault().exists) {
          return false;
        }
        restoreDefault(prior);
        return true;
      }
    );
  }, [accountScope, defaultUndoKey, key, offerUndo, ownerId, readDefault, restoreDefault, settingName, t, valueText]);

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
            onFollowDefault: followDefault,
            onUseAsDefault: promoteToDefault,
            state: 'pinned',
          }
        : undefined;
    }
    return defaultValue !== builtIn ? { onResetToBuiltIn: resetToBuiltIn, state: 'default' } : undefined;
  }, [
    builtIn,
    defaultValue,
    defaultValueText,
    followDefault,
    hasDefault,
    isPinMode,
    pinned,
    resetToBuiltIn,
    t,
    promoteToDefault,
  ]);

  return { disabled, layer, set, value };
};
