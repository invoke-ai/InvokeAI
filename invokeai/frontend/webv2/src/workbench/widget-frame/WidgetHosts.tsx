import { getWidgetHosts } from '@workbench/widgetRegistry';
import { Suspense, use } from 'react';

import { WidgetFailureBoundary } from './WidgetFailureBoundary';

const WidgetHost = ({ widget }: { widget: ReturnType<typeof getWidgetHosts>[number] }) => {
  const Host = use(widget.implementation.load()).host;

  return Host ? <Host /> : null;
};

const WidgetHostBoundary = ({ widget }: { widget: ReturnType<typeof getWidgetHosts>[number] }) => {
  const content = (
    <Suspense fallback={null}>
      <WidgetHost widget={widget} />
    </Suspense>
  );

  return widget.manifest.failurePolicy.isolateRenderFailure ? (
    <WidgetFailureBoundary
      resetKey={widget.manifest.id}
      widgetId={widget.manifest.id}
      onRetry={widget.implementation.retry}
    >
      {content}
    </WidgetFailureBoundary>
  ) : (
    content
  );
};

export const WidgetHosts = () => (
  <>
    {getWidgetHosts().map((widget) => (
      <WidgetHostBoundary key={widget.manifest.id} widget={widget} />
    ))}
  </>
);
