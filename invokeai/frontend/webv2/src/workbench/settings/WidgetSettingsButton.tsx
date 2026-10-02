import type { WidgetHeaderActions, WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Icon, Popover, Portal, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { RetryBoundary } from '@platform/ui/RetryBoundary';
import { Scrollable } from '@platform/ui/Scrollable';
import { resolveSettingsText } from '@platform/ui/settings/contracts';
import { Tooltip } from '@platform/ui/Tooltip';
import { createDeferredResource } from '@workbench/deferredResource';
import { useActiveProjectId, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { SettingsIcon } from 'lucide-react';
import { Suspense, use, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { openWorkbenchSettings } from './settingsDialogStore';

const quickResource = createDeferredResource(() => import('./WidgetQuickSettings'));
const QuickSettings = (props: WidgetViewProps) => {
  const { default: Content } = use(quickResource.load());
  return <Content {...props} />;
};

const QUICK_POSITIONING = { placement: 'bottom-end' } as const;
type WidgetSettingsButtonProps = WidgetViewProps & { SettingsActions?: WidgetHeaderActions };

/** Changing the project or instance disposes any open popover and its deferred content. */
export const WidgetSettingsButton = (props: WidgetSettingsButtonProps) => {
  const projectId = useActiveProjectId();
  return <TargetedWidgetSettingsButton key={`${projectId}:${props.instance.id}`} projectId={projectId} {...props} />;
};

/** All widget placements share one gear and one popover-to-dialog handoff. */
const TargetedWidgetSettingsButton = ({
  SettingsActions,
  projectId,
  ...props
}: WidgetSettingsButtonProps & { projectId: string }) => {
  const { t } = useTranslation();
  const [presentation, setPresentation] = useState<'closed' | 'quick' | 'dialog'>('closed');
  const trigger = useRef<HTMLButtonElement>(null);
  const queries = useWorkbenchQueries();
  const contribution = props.manifest.settings;
  const sectionId = contribution?.id;
  const instanceId = props.instance.id;
  const typeId = props.manifest.id;
  const isTargetCurrent = useCallback(
    () =>
      queries.isActiveProject(projectId) &&
      queries.getProject(projectId)?.widgetInstances[instanceId]?.typeId === typeId,
    [queries, projectId, instanceId, typeId]
  );
  const openDialog = useCallback(() => {
    if (!sectionId || !isTargetCurrent()) {
      return;
    }
    // A controlled close may also emit onOpenChange(false); it must retain this handoff state.
    setPresentation('dialog');
    openWorkbenchSettings({ sectionId, target: { projectId, instanceId } }, trigger.current ?? undefined);
  }, [sectionId, isTargetCurrent, projectId, instanceId]);
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (open) {
        if (isTargetCurrent()) {
          setPresentation('quick');
        }
      } else {
        setPresentation((current) => (current === 'dialog' ? current : 'closed'));
      }
    },
    [isTargetCurrent]
  );
  const getReturnFocus = useCallback(() => trigger.current, []);
  const loading = useMemo(
    () => (
      <Text role="status" fontSize="xs" color="fg.muted">
        {t('common.loading')}
      </Text>
    ),
    [t]
  );

  if (!contribution) {
    return null;
  }
  const label = resolveSettingsText(contribution.label, t);
  if (!contribution.quick?.length) {
    return (
      <Tooltip content={t('widgets.settingsLabel', { label })}>
        <IconButton
          ref={trigger}
          aria-label={t('widgets.settingsLabel', { label })}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={openDialog}
        >
          <Icon as={SettingsIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    );
  }
  return (
    <Popover.Root
      open={presentation === 'quick'}
      lazyMount
      unmountOnExit
      positioning={QUICK_POSITIONING}
      onOpenChange={handleOpenChange}
      restoreFocus={presentation !== 'dialog'}
      finalFocusEl={getReturnFocus}
    >
      <Popover.Trigger asChild>
        <IconButton
          ref={trigger}
          aria-label={t('widgets.settingsLabel', { label })}
          color="fg.muted"
          size="2xs"
          variant="ghost"
        >
          <Icon as={SettingsIcon} boxSize="3.5" />
        </IconButton>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="18rem">
            <Scrollable maxH="min(38rem, calc(100dvh - 4rem))">
              <Popover.Body p="2.5">
                <Popover.Title fontSize="xs" fontWeight="600" mb="1">
                  {t('widgets.settingsLabel', { label })}
                </Popover.Title>
                <RetryBoundary
                  retry={quickResource.retry}
                  message={t('settingsDialog.loadFailed')}
                  retryLabel={t('common.retry')}
                >
                  <Suspense fallback={loading}>
                    <QuickSettings {...props} />
                  </Suspense>
                </RetryBoundary>
                {SettingsActions ? (
                  <Box pt="2">
                    <SettingsActions {...props} />
                  </Box>
                ) : null}
                <Button size="xs" variant="ghost" w="full" mt="2" onClick={openDialog}>
                  {t('settingsDialog.allWidgetSettings', { widget: label })}
                </Button>
              </Popover.Body>
            </Scrollable>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
