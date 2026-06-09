import type { RegisteredWidget, WidgetViewProps } from '../types';
import { WidgetFailureBoundary } from './WidgetFailureBoundary';

interface WidgetRendererProps extends Omit<WidgetViewProps, 'manifest'> {
  widget: RegisteredWidget;
}

export const WidgetRenderer = ({ presentation, region, widget }: WidgetRendererProps) => {
  const View = widget.manifest.view;

  if (!View) {
    return null;
  }

  const content = <View manifest={widget.manifest} presentation={presentation} region={region} />;

  if (!widget.manifest.failurePolicy.isolateRenderFailure) {
    return content;
  }

  return <WidgetFailureBoundary widgetId={widget.manifest.id}>{content}</WidgetFailureBoundary>;
};
