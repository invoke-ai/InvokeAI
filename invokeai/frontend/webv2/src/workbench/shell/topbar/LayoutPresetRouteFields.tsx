import type { GraphWidgetSource } from '@workbench/graphWidgets';
import type { InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';
import type { LayoutPresetRoute } from '@workbench/layoutContracts';

import { createListCollection, HStack, Stack, Text } from '@chakra-ui/react';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { getNaturalDestination } from '@workbench/graphWidgets';
import { WidgetIcon } from '@workbench/iconResolver';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { RoutingDestinationSegments } from './RoutingDestinationSegments';

const AUTO_VALUE = 'auto';

type SourceSelectItem = { label: string; source: GraphWidgetSource | null; value: string };

const renderSourceOption = (item: SourceSelectItem) => (
  <HStack gap="1.5">
    {item.source ? <WidgetIcon boxSize="3.5" icon={getWidgetById(item.source.typeId)?.manifest.icon} /> : null}
    <Text as="span">{item.label}</Text>
  </HStack>
);

/** A workspace's route: Auto, where it may have none, or a source it places and where that source's results go. */
export const LayoutPresetRouteFields = ({
  allowsAuto,
  onChange,
  route,
  sourceOptions,
}: {
  allowsAuto: boolean;
  onChange: (route: LayoutPresetRoute | null) => void;
  /** Already resolved against the options (`resolveLayoutPresetRoute`). */
  route: LayoutPresetRoute | null;
  sourceOptions: readonly GraphWidgetSource[];
}) => {
  const { t } = useTranslation();
  const collection = useMemo(
    () =>
      createListCollection<SourceSelectItem>({
        items: [
          ...(allowsAuto ? [{ label: t('topbar.presets.autoRoute'), source: null, value: AUTO_VALUE }] : []),
          ...sourceOptions.map((source) => ({ label: source.label, source, value: source.sourceId })),
        ],
      }),
    [allowsAuto, sourceOptions, t]
  );
  const value = useMemo(() => [route?.sourceId ?? AUTO_VALUE], [route?.sourceId]);
  const selected = collection.items.find((item) => item.value === value[0]);
  const triggerProps = useMemo(() => ({ 'aria-label': t('topbar.presets.defaultSource') }), [t]);
  const handleSourceChange = useCallback(
    (event: { value: string[] }) => {
      const sourceId = event.value[0];

      if (sourceId === AUTO_VALUE) {
        onChange(null);
      } else if (sourceId) {
        onChange({
          destination: route?.destination ?? getNaturalDestination(sourceId as InvocationSourceId),
          sourceId: sourceId as InvocationSourceId,
        });
      }
    },
    [onChange, route?.destination]
  );
  const handleDestinationChange = useCallback(
    (destination: ResultDestination) => {
      if (route) {
        onChange({ ...route, destination });
      }
    },
    [onChange, route]
  );

  return (
    <Stack gap="4">
      <Field
        helpText={
          sourceOptions.length === 0
            ? t('topbar.presets.noInvocationSource')
            : route === null
              ? t('topbar.presets.autoRouteHint')
              : undefined
        }
        label={t('topbar.presets.defaultSource')}
      >
        <Select
          collection={collection}
          renderItem={renderSourceOption}
          triggerProps={triggerProps}
          value={value}
          valueText={selected ? renderSourceOption(selected) : undefined}
          onValueChange={handleSourceChange}
        />
      </Field>
      <Field disabled={!route} label={t('topbar.presets.defaultDestination')}>
        <RoutingDestinationSegments
          ariaLabel={t('topbar.presets.defaultDestination')}
          disabled={!route}
          value={route?.destination ?? null}
          onChange={handleDestinationChange}
        />
      </Field>
    </Stack>
  );
};
