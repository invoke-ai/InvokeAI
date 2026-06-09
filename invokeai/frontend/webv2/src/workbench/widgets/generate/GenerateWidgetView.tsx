import { FieldPlaceholder, GraphBearingWidgetHeader, WidgetPanelFrame } from '../../components/WidgetFrames';
import { WidgetFailureBoundary } from '../../components/WidgetFailureBoundary';
import type { WidgetViewProps } from '../../types';

export const GenerateWidgetView = ({ manifest }: WidgetViewProps) => (
  <WidgetFailureBoundary widgetId="generate">
    <WidgetPanelFrame region="left">
      <GraphBearingWidgetHeader manifest={manifest} region="left" />
      <FieldPlaceholder label="Prompt" h="6rem" />
      <FieldPlaceholder label="Negative prompt" h="3rem" />
      <FieldPlaceholder label="Model" h="2.25rem" />
      <FieldPlaceholder label="Dimensions" h="2.25rem" />
    </WidgetPanelFrame>
  </WidgetFailureBoundary>
);
