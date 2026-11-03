import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { Project } from '@workbench/projectContracts';

import { Box, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { RetryBoundary } from '@platform/ui/RetryBoundary';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { useOptionalWorkbenchSelector } from '@workbench/WorkbenchContext';
import { Suspense, use, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { SettingsEntry, SettingsSection } from './catalog';

import { PROJECT_SETTINGS_OWNER } from './projectSettings';
import { SettingsScopeLabel } from './SettingsScopeLabel';

const FILL_PROPS = { display: 'flex', flex: '1', flexDirection: 'column', minH: '0' } as const;

/**
 * What a section's fields edit in the open project: a requested target as given, else the project and the section's
 * widget. A requested target that is gone stays as it was, so its controls disable rather than move to another project
 * or instance. Undefined (the user's defaults) where the project has no widget to pin to.
 */
const resolveSettingsTarget = (
  project: Project,
  section: SettingsSection,
  requested: SettingsTarget | undefined
): SettingsTarget | undefined => {
  // The binding checks that an explicit instance still exists and is the section's widget.
  if (requested && (requested.projectId !== project.id || requested.instanceId)) {
    return requested;
  }
  // The project's own settings (the Generate widget's section) pin on the project, whichever widgets it has.
  if (!section.widgetId || section.widgetId === PROJECT_SETTINGS_OWNER) {
    return { projectId: project.id };
  }
  const instance = getProjectWidgetInstance(project, section.widgetId);
  return instance ? { projectId: project.id, instanceId: instance.id } : undefined;
};

/** Whether a section's per-project fields edit the user's defaults here: no project, or no widget in it to pin to. */
export const useEditsDefaults = (section: SettingsSection): boolean =>
  useOptionalWorkbenchSelector(
    (snapshot) =>
      Boolean(
        section.widgetId &&
        section.widgetId !== PROJECT_SETTINGS_OWNER &&
        !getProjectWidgetInstance(snapshot.activeProject, section.widgetId)
      ),
    true
  );

const LoadedSetting = ({
  entry,
  fill,
  target,
  surface,
  isRevealed,
  onReveal,
  onRevealed,
}: {
  entry: SettingsEntry;
  fill: boolean;
  target?: SettingsTarget;
  surface: 'quick' | 'dialog';
  isRevealed: boolean;
  onReveal?: (sectionId: string, entryId: string) => void;
  onRevealed?: () => void;
}) => {
  const { Field } = use(entry.resource.load());
  const attach = useCallback(
    (element: HTMLDivElement | null) => {
      if (!element || !isRevealed) {
        return;
      }
      // The control, not the layer indicator that precedes it after the label.
      const control = element.querySelector<HTMLElement>('input,button:not([data-setting-layer]),[tabindex="0"]');
      element.scrollIntoView({ block: 'nearest' });
      control?.focus({ preventScroll: true });
      onRevealed?.();
    },
    [isRevealed, onRevealed]
  );
  return (
    <Box ref={attach} {...(fill ? FILL_PROPS : undefined)}>
      <Field field={entry.field} surface={surface} target={target} onReveal={onReveal} />
    </Box>
  );
};

/** A custom editor that owns its scrolling fills the dialog body instead of stacking in it. */
export const isFillSettingsEntry = (entry: SettingsEntry, surface: 'quick' | 'dialog'): boolean =>
  surface === 'dialog' && entry.field.kind === 'custom' && entry.field.fill === true;

export const SettingsEntryView = ({
  entry,
  section,
  target: requestedTarget,
  surface = 'dialog',
  search = false,
  onReveal,
  revealEntryId,
  onRevealed,
  showGroup = false,
}: {
  entry: SettingsEntry;
  section: SettingsSection;
  target?: SettingsTarget;
  surface?: 'quick' | 'dialog';
  search?: boolean;
  onReveal?: (sectionId: string, entryId: string) => void;
  /** The entry to scroll to and focus once it loads; `onRevealed` then clears the request. */
  revealEntryId?: string;
  onRevealed?: () => void;
  showGroup?: boolean;
}) => {
  const { t } = useTranslation();
  const target = useOptionalWorkbenchSelector<SettingsTarget | undefined>(
    (snapshot) => resolveSettingsTarget(snapshot.activeProject, section, requestedTarget),
    undefined
  );
  const reveal = useCallback(() => onReveal?.(section.id, entry.field.id), [onReveal, section.id, entry.field.id]);
  const loading = useMemo(
    () => (
      <Text role="status" fontSize="md" color="fg.muted">
        {t('common.loading')}
      </Text>
    ),
    [t]
  );
  const scope = entry.field.scope;
  const isDestination = search && entry.field.kind === 'custom';
  const fill = !search && isFillSettingsEntry(entry, surface);
  const framed = !search && entry.field.kind === 'custom' && entry.field.framed === true;
  return (
    <>
      {showGroup && entry.field.group ? (
        <Text
          as="h3"
          pt={surface === 'quick' ? '2' : '5'}
          pb="1"
          fontWeight="600"
          fontSize={surface === 'quick' ? 'xs' : 'md'}
          color="fg.muted"
        >
          {resolveSettingsText(entry.field.group, t)}
        </Text>
      ) : null}
      <Box
        data-setting-id={entry.field.id}
        py={surface === 'quick' ? '1.5' : '4'}
        borderBottomWidth={fill || framed ? '0' : '1px'}
        borderColor="border.subtle"
        {...(fill ? FILL_PROPS : undefined)}
      >
        {isDestination ? (
          <Stack gap="2">
            <Text fontSize="lg" fontWeight="500">
              {resolveSettingsText(entry.field.label, t)}
            </Text>
            {entry.field.description ? (
              <Text fontSize="md" color="fg.muted">
                {resolveSettingsText(entry.field.description, t)}
              </Text>
            ) : null}
            <Button variant="outline" alignSelf="start" onClick={reveal}>
              {t('settingsDialog.openEditor')}
            </Button>
          </Stack>
        ) : (
          <RetryBoundary
            key={`${entry.field.id}:${target?.projectId ?? ''}:${target?.instanceId ?? ''}`}
            retry={entry.resource.retry}
            message={t('settingsDialog.loadFailed')}
            retryLabel={t('common.retry')}
          >
            <Suspense fallback={loading}>
              <LoadedSetting
                entry={entry}
                fill={fill}
                isRevealed={revealEntryId === entry.field.id}
                target={target}
                surface={surface}
                onReveal={onReveal}
                onRevealed={onRevealed}
              />
            </Suspense>
          </RetryBoundary>
        )}
        {surface === 'dialog' && scope !== section.entries[0]?.field.scope ? (
          <Box mt="2">
            <SettingsScopeLabel scope={scope} />
          </Box>
        ) : null}
        {search && !isDestination ? (
          <Button size="sm" variant="ghost" mt="1" onClick={reveal}>
            {t('settingsDialog.showInSection')}
          </Button>
        ) : null}
      </Box>
    </>
  );
};
