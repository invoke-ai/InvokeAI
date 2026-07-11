import type { FilterOperationSessionState } from '@workbench/widgets/layers/filterOperationSession';

import { HStack, Menu, Portal, Text } from '@chakra-ui/react';
import { Button, MenuContent } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import {
  buildFilterDefaults,
  getFilterDefinition,
  isFilterConfigValid,
} from '@workbench/generation/canvas/filterGraphs';
import { useFilterSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { LayerFilterControls } from '@workbench/widgets/layers/LayerFilterControls';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

export interface FilterActionEligibility {
  canApply: boolean;
  canCancel: boolean;
  canEdit: boolean;
  canProcess: boolean;
  canReset: boolean;
  canSave: boolean;
}

export const getFilterActionEligibility = (
  session: FilterOperationSessionState,
  isExternalInteractionLocked = false
): FilterActionEligibility => {
  const busy = session.status === 'processing' || session.status === 'committing';
  const actionsBlocked = busy || isExternalInteractionLocked;
  const hasPreview = session.preview !== null && !actionsBlocked;
  const isValid = isFilterConfigValid(session.draft.type, session.draft.settings);
  return {
    canApply: hasPreview,
    canCancel: true,
    canEdit: !actionsBlocked,
    canProcess: !actionsBlocked && isValid,
    canReset: !actionsBlocked,
    canSave: hasPreview,
  };
};

export const FilterOptions = ({
  engine,
  isExternalInteractionLocked = false,
}: ToolOptionsComponentProps & { isExternalInteractionLocked?: boolean }) => {
  const { t } = useTranslation();
  const session = useFilterSession(engine);
  const setType = useCallback(
    (type: string) => {
      const definition = getFilterDefinition(type);
      engine.updateFilterOperation({ settings: definition ? buildFilterDefaults(definition) : {}, type });
    },
    [engine]
  );
  const setSettings = useCallback(
    (settings: Record<string, unknown>) => {
      const current = engine.stores.filterSession.get();
      if (current) {
        engine.updateFilterOperation({ settings, type: current.draft.type });
      }
    },
    [engine]
  );
  const process = useCallback(() => void engine.processFilterOperation(), [engine]);
  const reset = useCallback(() => {
    const current = engine.stores.filterSession.get();
    if (!current) {
      return;
    }
    const definition = getFilterDefinition(current.draft.type);
    engine.resetFilterOperation(definition ? buildFilterDefaults(definition) : {});
  }, [engine]);
  const apply = useCallback(() => void engine.commitFilterOperation('apply', makeImageDurable), [engine]);
  const saveRaster = useCallback(() => void engine.commitFilterOperation('raster', makeImageDurable), [engine]);
  const saveControl = useCallback(() => void engine.commitFilterOperation('control', makeImageDurable), [engine]);
  const cancel = useCallback(() => engine.cancelFilterOperation(), [engine]);

  if (!session) {
    return null;
  }
  const eligibility = getFilterActionEligibility(session, isExternalInteractionLocked);

  return (
    <HStack align="center" gap="2" maxW="calc(100vw - 2rem)" overflowX="auto">
      <LayerFilterControls
        disabled={!eligibility.canEdit}
        filterType={session.draft.type}
        focusFilter={false}
        settings={session.draft.settings}
        onFilterTypeChange={setType}
        onSettingsChange={setSettings}
      />
      <Button disabled={!eligibility.canProcess} loading={session.status === 'processing'} size="xs" onClick={process}>
        {t('widgets.layers.selectObject.process')}
      </Button>
      <Button disabled={!eligibility.canReset} size="xs" variant="ghost" onClick={reset}>
        {t('widgets.layers.selectObject.reset')}
      </Button>
      <Button disabled={!eligibility.canApply} loading={session.status === 'committing'} size="xs" onClick={apply}>
        {t('common.apply')}
      </Button>
      <Menu.Root>
        <Menu.Trigger asChild>
          <Button disabled={!eligibility.canSave} size="xs" variant="ghost">
            {t('widgets.layers.selectObject.saveAs')} <ChevronDownIcon size={12} />
          </Button>
        </Menu.Trigger>
        <Portal>
          <Menu.Positioner>
            <MenuContent minW="11rem" py="1">
              <Menu.Item value="raster" onClick={saveRaster}>
                <Menu.ItemText>{t('widgets.layers.selectObject.saveAs_raster')}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="control" onClick={saveControl}>
                <Menu.ItemText>{t('widgets.layers.selectObject.saveAs_control')}</Menu.ItemText>
              </Menu.Item>
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Button disabled={!eligibility.canCancel} size="xs" variant="ghost" onClick={cancel}>
        {t('common.cancel')}
      </Button>
      {session.error ? (
        <Text color="fg.error" fontSize="2xs" role="alert">
          {session.error}
        </Text>
      ) : null}
    </HStack>
  );
};
