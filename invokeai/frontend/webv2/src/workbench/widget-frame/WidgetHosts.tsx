import { getWidgetHosts } from '@workbench/widgetRegistry';
import { Suspense, use } from 'react';

import { WidgetFailureBoundary } from './WidgetFailureBoundary';

const WidgetHost = ({ widget }: { widget: ReturnType<typeof getWidgetHosts>[number] }) => {
  const Host = use(widget.host!.load());

  // react-compiler flags any JSX tag that is directly the value returned by
  // `use()` as though it were freshly created each render. `Host` is the
  // module's cached export, resolved once by the deferred resource and
  // stable across renders; the false positive disappears the moment the
  // value is read through a property access instead of being the call's
  // direct result, which is what the sibling `WidgetRenderer` slots do.
  // eslint-disable-next-line react/react-compiler
  return <Host />;
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
      widget={widget}
      widgetId={widget.manifest.id}
      onRetry={widget.host!.retry}
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
