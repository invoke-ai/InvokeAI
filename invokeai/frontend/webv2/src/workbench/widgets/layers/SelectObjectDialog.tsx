import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { SamModel } from '@workbench/generation/canvas/samGraph';
import type { ChangeEvent, FormEvent } from 'react';

import { chakra, createListCollection, Dialog, Input, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import { socketHub } from '@workbench/backend/socketHub';
import { runUtilityGraph } from '@workbench/canvas-engine/backend/utilityQueue';
import { Button, CloseButton, Field, Select } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import { useNotify } from '@workbench/useNotify';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { createLayerActionSession } from './layerActionSession';
import {
  createDefaultSelectObjectOptions,
  isSelectObjectPromptValid,
  routeSelectObjectResult,
  runSelectObject,
  type SelectObjectRouteResult,
  type SelectObjectRunResult,
  type SelectObjectTarget,
} from './layerImageResult';

interface SelectObjectDialogProps {
  engine: CanvasEngine | null;
  isOpen: boolean;
  layerId: string;
  onClose(): void;
}

const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;

const messageOf = (result: { message: string }): string => result.message;

export const SelectObjectDialog = ({ engine, isOpen, layerId, onClose }: SelectObjectDialogProps) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const defaults = createDefaultSelectObjectOptions();
  const [session] = useState(createLayerActionSession);
  const [prompt, setPrompt] = useState(defaults.prompt);
  const [model, setModel] = useState<SamModel>(defaults.model);
  const [applyPolygonRefinement, setApplyPolygonRefinement] = useState(defaults.applyPolygonRefinement);
  const [target, setTarget] = useState<SelectObjectTarget>(defaults.target);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const modelCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: SamModel }>({
        items: [
          { label: t('widgets.layers.selectObject.modelSam2Large'), value: 'segment-anything-2-large' },
          { label: t('widgets.layers.selectObject.modelHuge'), value: 'segment-anything-huge' },
        ],
      }),
    [t]
  );
  const targetCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: SelectObjectTarget }>({
        items: [
          { label: t('widgets.layers.selectObject.targetSelection'), value: 'selection' },
          { label: t('widgets.layers.selectObject.targetInpaintMask'), value: 'inpaint_mask' },
          { label: t('widgets.layers.selectObject.targetRegionalGuidance'), value: 'regional_guidance' },
        ],
      }),
    [t]
  );
  const modelValue = useMemo(() => [model], [model]);
  const targetValue = useMemo(() => [target], [target]);

  const reset = useCallback(() => {
    const next = createDefaultSelectObjectOptions();
    setPrompt(next.prompt);
    setModel(next.model);
    setApplyPolygonRefinement(next.applyPolygonRefinement);
    setTarget(next.target);
    setError(null);
    setIsRunning(false);
  }, []);

  const close = useCallback(() => {
    session.cancel();
    reset();
    onClose();
  }, [onClose, reset, session]);

  const contentRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node) {
        return;
      }
      return () => {
        session.cancel();
      };
    },
    [session]
  );

  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        close();
      }
    },
    [close]
  );
  const handlePromptChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setPrompt(event.currentTarget.value);
    setError(null);
  }, []);
  const handleModelChange = useCallback(({ value }: SelectValueChangeDetails) => {
    const next = value[0] as SamModel | undefined;
    if (next) {
      setModel(next);
    }
  }, []);
  const handleTargetChange = useCallback(({ value }: SelectValueChangeDetails) => {
    const next = value[0] as SelectObjectTarget | undefined;
    if (next) {
      setTarget(next);
    }
  }, []);
  const handleRefinementChange = useCallback(({ checked }: { checked: boolean }) => {
    setApplyPolygonRefinement(checked);
  }, []);

  const showRunFailure = useCallback(
    (result: Exclude<SelectObjectRunResult, { status: 'ready' }>) => {
      switch (result.status) {
        case 'aborted':
          setError(t('widgets.layers.selectObject.aborted'));
          return;
        case 'missing':
        case 'disabled':
        case 'unsupported':
        case 'empty':
        case 'not-ready':
          setError(
            t(
              result.status === 'not-ready'
                ? 'widgets.layers.selectObject.notReady'
                : `widgets.layers.selectObject.${result.status}`
            )
          );
          return;
        case 'failed':
          setError(t('widgets.layers.selectObject.failed', { message: messageOf(result) }));
          return;
      }
    },
    [t]
  );

  const showRouteResult = useCallback(
    (result: SelectObjectRouteResult, requestedTarget: SelectObjectTarget): boolean => {
      switch (result.status) {
        case 'selected':
          notify.success(t('widgets.layers.selectObject.selectionSuccess'));
          return true;
        case 'committed':
          notify.success(
            t(
              requestedTarget === 'inpaint_mask'
                ? 'widgets.layers.selectObject.inpaintMaskSuccess'
                : 'widgets.layers.selectObject.regionalGuidanceSuccess'
            )
          );
          return true;
        case 'aborted':
          setError(t('widgets.layers.selectObject.aborted'));
          return false;
        case 'missing':
        case 'stale':
          setError(t('widgets.layers.selectObject.stale'));
          return false;
        case 'locked':
        case 'unsupported':
        case 'busy':
          setError(t(`widgets.layers.selectObject.${result.status}`));
          return false;
        case 'failed':
          setError(
            t(
              requestedTarget === 'selection'
                ? 'widgets.layers.selectObject.failed'
                : 'widgets.layers.selectObject.durabilityFailure',
              { message: messageOf(result) }
            )
          );
          return false;
      }
    },
    [notify, t]
  );

  const runContext = useMemo(
    () => ({
      applyPolygonRefinement,
      close,
      engine,
      layerId,
      model,
      prompt,
      session,
      showRouteResult,
      showRunFailure,
      target,
    }),
    [applyPolygonRefinement, close, engine, layerId, model, prompt, session, showRouteResult, showRunFailure, target]
  );

  const run = useCallback(async (): Promise<void> => {
    const {
      applyPolygonRefinement,
      close,
      engine,
      layerId,
      model,
      prompt,
      session,
      showRouteResult,
      showRunFailure,
      target,
    } = runContext;
    if (!engine || !isSelectObjectPromptValid(prompt)) {
      return;
    }
    const request = session.begin();
    if (!request) {
      return;
    }
    const requestedTarget = target;
    setError(null);
    setIsRunning(true);
    try {
      const executorDeps = engine.getCompositeExecutorDeps();
      const result = await runSelectObject({
        applyPolygonRefinement,
        deps: {
          exportLayer: () => engine.exportBakedLayerBlob(layerId, { includeDisabled: true }),
          runGraph: (options) => runUtilityGraph({ ...options, hub: socketHub }),
          uploadIntermediate: async (blob, signal) => {
            if (signal?.aborted) {
              throw new DOMException('Object selection aborted', 'AbortError');
            }
            const uploaded = await executorDeps.uploadImage(blob);
            if (signal?.aborted) {
              throw new DOMException('Object selection aborted', 'AbortError');
            }
            return { imageName: uploaded.imageName };
          },
        },
        layerId,
        model,
        prompt,
        signal: request.signal,
      });
      if (!session.isCurrent(request.token)) {
        return;
      }
      if (result.status !== 'ready') {
        showRunFailure(result);
        return;
      }
      const routed = await routeSelectObjectResult({
        deps: {
          commitMaskImageResult: (options) => engine.commitMaskImageResult(options),
          makeImageDurable,
          replaceSelectionFromImage: (guard, image, rect, signal) =>
            engine.replaceSelectionFromImage(guard, image, rect, signal),
        },
        isCurrent: () => session.isCurrent(request.token),
        result,
        signal: request.signal,
        target: requestedTarget,
      });
      if (!session.isCurrent(request.token)) {
        return;
      }
      if (showRouteResult(routed, requestedTarget)) {
        close();
      }
    } finally {
      if (session.isCurrent(request.token)) {
        setIsRunning(false);
      }
      session.finish(request.token);
    }
  }, [runContext]);

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>): void => {
      event.preventDefault();
      void run();
    },
    [run]
  );

  const canRun = engine !== null && !isRunning && isSelectObjectPromptValid(prompt);

  return (
    <Dialog.Root lazyMount open={isOpen} placement="center" size="sm" unmountOnExit onOpenChange={handleOpenChange}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content ref={contentRef} bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg">
            <chakra.form onSubmit={handleSubmit}>
              <Dialog.Header>
                <Dialog.Title fontSize="sm" fontWeight="700">
                  {t('widgets.layers.selectObject.title')}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="4">
                  <Field id="select-object-prompt" label={t('widgets.layers.selectObject.prompt')} required>
                    <Input
                      autoFocus
                      aria-label={t('widgets.layers.selectObject.prompt')}
                      autoComplete="off"
                      disabled={isRunning}
                      name="selectObjectPrompt"
                      placeholder={t('widgets.layers.selectObject.promptPlaceholder')}
                      size="sm"
                      value={prompt}
                      onChange={handlePromptChange}
                    />
                  </Field>
                  <Field label={t('widgets.layers.selectObject.model')}>
                    <Select
                      aria-label={t('widgets.layers.selectObject.model')}
                      collection={modelCollection}
                      disabled={isRunning}
                      positioning={SELECT_POSITIONING}
                      size="sm"
                      value={modelValue}
                      onValueChange={handleModelChange}
                    />
                  </Field>
                  <Field label={t('widgets.layers.selectObject.target')}>
                    <Select
                      aria-label={t('widgets.layers.selectObject.target')}
                      collection={targetCollection}
                      disabled={isRunning}
                      positioning={SELECT_POSITIONING}
                      size="sm"
                      value={targetValue}
                      onValueChange={handleTargetChange}
                    />
                  </Field>
                  <Switch.Root
                    checked={applyPolygonRefinement}
                    disabled={isRunning}
                    size="sm"
                    onCheckedChange={handleRefinementChange}
                  >
                    <Switch.HiddenInput />
                    <Switch.Control>
                      <Switch.Thumb />
                    </Switch.Control>
                    <Switch.Label>
                      <Text fontSize="xs">{t('widgets.layers.selectObject.refine')}</Text>
                    </Switch.Label>
                  </Switch.Root>
                  {error ? (
                    <Text color="fg.error" fontSize="xs" role="alert">
                      {error}
                    </Text>
                  ) : null}
                  {isRunning ? (
                    <Text color="fg.muted" fontSize="xs" role="status">
                      {t('widgets.layers.selectObject.running')}
                    </Text>
                  ) : null}
                </Stack>
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button size="xs" type="button" variant="ghost" onClick={close}>
                  {t('widgets.layers.selectObject.cancel')}
                </Button>
                <Button disabled={!canRun} loading={isRunning} size="xs" type="submit" variant="solid">
                  {t('widgets.layers.selectObject.run')}
                </Button>
              </Dialog.Footer>
            </chakra.form>
            <Dialog.CloseTrigger asChild>
              <CloseButton aria-label={t('widgets.layers.selectObject.cancel')} color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
