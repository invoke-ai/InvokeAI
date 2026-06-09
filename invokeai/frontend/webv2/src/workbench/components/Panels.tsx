import { Stack, Text } from '@chakra-ui/react';

import type { WidgetId, WorkbenchRegion } from '../types';
import { getWidgetById } from '../widgetRegistry';
import { WidgetRenderer } from './WidgetRenderer';

/** Left panel — hosts the active registered widget panel view. */
export const LeftPanel = ({ widgetId }: { widgetId: WidgetId }) => (
  <WidgetPanelSlot widgetId={widgetId} panel="leftPanel" />
);

/** Right panel — hosts the active registered widget panel view. */
export const RightPanel = ({ widgetId }: { widgetId: WidgetId }) => (
  <WidgetPanelSlot widgetId={widgetId} panel="rightPanel" />
);

const panelRegions = {
  leftPanel: 'left',
  rightPanel: 'right',
} as const satisfies Record<string, WorkbenchRegion>;

const WidgetPanelSlot = ({ widgetId, panel }: { widgetId: WidgetId; panel: keyof typeof panelRegions }) => {
  const widget = getWidgetById(widgetId);
  const View = widget?.manifest.view;
  const region = panelRegions[panel];

  if (!widget || widget.status !== 'enabled' || !View) {
    return <MissingWidgetPanel label={widget?.manifest.labelText ?? widgetId} />;
  }

  return <WidgetRenderer widget={widget} region={region} />;
};

const MissingWidgetPanel = ({ label }: { label: string }) => (
  <Stack
    as="aside"
    bg="bg.shell"
    borderColor="border.subtle"
    borderRightWidth="1px"
    color="fg.subtle"
    flexShrink={0}
    gap="2"
    p="3"
    w="16rem"
  >
    <Text fontSize="xs" fontWeight="700">
      {label}
    </Text>
    <Text fontSize="2xs">Widget view unavailable.</Text>
  </Stack>
);
