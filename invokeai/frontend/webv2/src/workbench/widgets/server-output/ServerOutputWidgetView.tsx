import type { WidgetInstanceRuntimeMeta, WidgetRuntimeApi, WidgetViewProps } from '@workbench/widgetContracts';

import { useMountEffect } from '@platform/react/useMountEffect';
import { refetchFrontendConfig } from '@workbench/frontendConfig';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { readWidgetCapabilities } from '@workbench/widgetCapabilities';
import { SquareTerminalIcon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { createServerOutputController, serverOutputControllers } from './controller';
import { ServerOutputPanel } from './ServerOutputPanel';
import { ServerOutputLoading, ServerOutputRefusal } from './ServerOutputStates';
import { getServerOutputRefusal } from './serverOutputStatus';
import { useServerOutputAccess, useServerOutputController } from './useServerOutputAccess';

/**
 * A saved layout restores this widget without discovery, so the expanded view checks the grant itself and mounts the
 * reader only once it holds; until then nothing reads the transcript. A refused read keeps its message until the
 * grant is checked again; a renewed grant mounts a fresh reader.
 */
const ServerOutputGate = ({
  instance,
  runtime,
}: {
  instance: WidgetInstanceRuntimeMeta;
  runtime: WidgetRuntimeApi;
}) => {
  const [controller] = useState(createServerOutputController);
  const snapshot = useServerOutputController(controller);
  const access = useServerOutputAccess(snapshot);
  const refusal = getServerOutputRefusal(access);
  const [isChecking, setIsChecking] = useState(false);

  useMountEffect(() => {
    serverOutputControllers.set(instance.id, controller);

    return () => {
      if (serverOutputControllers.get(instance.id) === controller) {
        serverOutputControllers.delete(instance.id);
      }
    };
  });

  const checkAgain = useCallback(() => {
    setIsChecking(true);
    void refetchFrontendConfig().finally(() => {
      setIsChecking(false);
      if (readWidgetCapabilities().viewServerOutput === 'granted') {
        controller.renew();
      }
    });
  }, [controller]);
  const remove = useCallback(() => runtime.workbench.closeWidgetInstance(instance.id), [instance.id, runtime]);

  if (refusal !== null) {
    return (
      <ServerOutputRefusal
        isChecking={isChecking}
        reason={refusal}
        onCheckAgain={access.canViewByRole ? checkAgain : undefined}
        onRemove={remove}
      />
    );
  }

  if (access.capability === 'pending') {
    return <ServerOutputLoading />;
  }

  return <ServerOutputPanel key={snapshot.generation} controller={controller} />;
};

export const ServerOutputWidgetView = ({ instance, presentation, region, runtime }: WidgetViewProps) => {
  const { t } = useTranslation();

  if (region === 'bottom' && presentation !== 'expanded') {
    return <StatusWidgetChip icon={SquareTerminalIcon}>{t('widgets.labels.serverOutput')}</StatusWidgetChip>;
  }

  return <ServerOutputGate instance={instance} runtime={runtime} />;
};
